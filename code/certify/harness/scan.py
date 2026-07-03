"""Whole-store SCAN over the compiled card store.

This is the contribution, NOT top-k retrieval. To certify absence we must look at
EVERY card, because a single un-inspected card could hold the evidence. We do it
in small windows so each LLM call stays well under the lost-in-middle floor:

  - completeness: every card is read (unlike RAG top-k -> can't certify "none")
  - semantic:     an LLM judges meaning (unlike grep -> paraphrase-blind)
  - no LITM floor: windows are small (unlike dump-all-N -> lost-in-middle)

Cost is O(N/window) LLM calls per query. That is the price of a real absence
certificate; it amortizes over a persistent store and the emitted certificate is
cheap to re-check. Windows run concurrently over the server's parallel slots.

Reduce rule: if ANY window reports a SUPPORT/CONTRADICT hit -> non-absent
(evidence exists). If every window says NONE -> NEI (certified absent, modulo the
measured extraction recall of the compile step).
"""
import re

SCAN_SYS = (
    "You are a precise evidence finder. You are given a CLAIM and a numbered list "
    "of findings from a scientific corpus. Decide whether ANY finding bears on the "
    "claim's truth. A finding SUPPORTS the claim if it agrees with it. A finding "
    "CONTRADICTS the claim if it states a different value, quantity, count, or "
    "direction than the claim asserts (e.g. claim says 'four' but a finding says "
    "'two' -> CONTRADICT). Be decisive: relevant evidence counts even if worded "
    "differently from the claim. "
    "If at least one finding bears on the claim, reply exactly: HIT <number> "
    "<SUPPORT|CONTRADICT>. If none are relevant, reply exactly: NONE. One line only."
)

_HIT = re.compile(r"HIT\s+(\d+)\s+(SUPPORT|CONTRADICT)", re.IGNORECASE)


def _window_prompt(claim, cards_window):
    lines = "\n".join(f"{i+1}. {c}" for i, c in enumerate(cards_window))
    return f"CLAIM: {claim}\n\nFINDINGS:\n{lines}\n\nAnswer (HIT <n> <SUPPORT|CONTRADICT>, or NONE):"


def method_scan(llm, claim, cards, window=30, max_new_tokens=16):
    """Read ALL cards in windows; certify absence iff every window says NONE."""
    ids = list(cards.keys())
    texts = [cards[d] for d in ids]
    windows = [texts[s:s + window] for s in range(0, len(texts), window)]
    prompts = [_window_prompt(claim, w) for w in windows]
    outs = llm.chat_batch(SCAN_SYS, prompts, max_new_tokens=max_new_tokens)

    label = "NEI"
    hit_window = None
    for wi, o in enumerate(outs):
        m = _HIT.search(o or "")
        if m:
            label = m.group(2).upper()
            hit_window = wi
            break  # first concrete hit decides non-absence
    # approx query cost: sum of all window prompt chars / 4 (every card was read)
    n_ctx = sum(len(p) for p in prompts) // 4
    return {"label": label, "n_ctx": n_ctx, "n_windows": len(windows),
            "n_cards": len(ids), "hit_window": hit_window}
