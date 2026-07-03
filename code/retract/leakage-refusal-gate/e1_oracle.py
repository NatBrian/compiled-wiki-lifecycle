"""E1 — Oracle-Matched Partial Deletion (ANCHOR experiment for C1).

Proves that partial_delete achieves oracle-level precision/recall when
removing a source Sk from a compiled wiki.  The oracle is defined by
recompiling the wiki from scratch on corpus \ {Sk}.

Pipeline:
  Step 1: Load compiled wiki from E0 (results/compiled_wiki.jsonl)
  Step 2: Select K=20 retraction candidates (stratified)
  Step 3: Build oracle wikis (recompile-without-Sk)
  Step 4: Run partial_delete (our method)
  Step 5: Run baselines (RAG-drop, Naive-wiki-delete, FLAT/param-unlearn)
  Step 6: Compute metrics (Oracle-F1, Residue-rate, Over-deletion-rate, ChangeAcc)
  Step 7: Save results

Usage:
  python e1_oracle.py --mode cpu --gpu-id 4
  python e1_oracle.py --mode gpu --gpu-id 4 --port 8102

Dependencies: rank_bm25, scipy, sklearn, sentence_transformers, transformers, numpy
Environment: ls_test conda env, vLLM 0.19.1, CUDA 12.6
"""
import argparse
import copy
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Deferred heavy imports (loaded once at runtime to avoid import-time GPU init)
# ---------------------------------------------------------------------------
_bge_model = None
_nli_pipeline = None
_llm_client = None


def _get_bge():
    global _bge_model
    if _bge_model is None:
        from sentence_transformers import SentenceTransformer
        _bge_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _bge_model


def _get_nli(mode: str):
    """Return NLI pipeline: DeBERTa in cpu mode, Qwen NLI in gpu mode."""
    global _nli_pipeline
    if _nli_pipeline is None:
        from transformers import pipeline
        if mode == "cpu":
            model_name = "cross-encoder/nli-deberta-v3-small"
        else:
            model_name = "cross-encoder/nli-deberta-v3-large"
        _nli_pipeline = pipeline("text-classification", model=model_name,
                                  truncation=True, max_length=512)
    return _nli_pipeline


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Default paths (relative to experiment root)
DEFAULT_WIKI_PATH = os.path.join(EXPERIMENT_ROOT, "results", "compiled_wiki.jsonl")
DEFAULT_CORPUS_DIR = os.path.join(EXPERIMENT_ROOT, "data", "wikipedia")
DEFAULT_RESULTS_DIR = os.path.join(EXPERIMENT_ROOT, "results")
DEFAULT_ORACLE_DIR = os.path.join(EXPERIMENT_ROOT, "results", "oracle_diffs")

K_CANDIDATES = 20
SIM_DIE_THRESHOLD = 0.70
SIM_SURVIVE_THRESHOLD = 0.85
BOOTSTRAP_ITERS = 1000
BONFERRONI_ALPHA = 0.017


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Claim:
    """A single claim in the compiled wiki."""

    __slots__ = ("claim_id", "text", "source_ids", "fusion_type", "sole_entailer")

    def __init__(self, claim_id: str, text: str, source_ids: List[str],
                 fusion_type: str, sole_entailer: Optional[str] = None):
        self.claim_id = claim_id
        self.text = text
        self.source_ids = list(source_ids)
        self.fusion_type = fusion_type
        self.sole_entailer = sole_entailer

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "source_ids": self.source_ids,
            "fusion_type": self.fusion_type,
            "sole_entailer": self.sole_entailer,
        }


class Wiki:
    """In-memory compiled wiki."""

    def __init__(self, claims: List[Claim]):
        self.claims: List[Claim] = list(claims)

    def remove_claim(self, claim: Claim):
        self.claims = [c for c in self.claims if c.claim_id != claim.claim_id]

    def get_texts(self) -> List[str]:
        return [c.text for c in self.claims]


# ---------------------------------------------------------------------------
# Step 1: Load compiled wiki
# ---------------------------------------------------------------------------

def load_compiled_wiki(wiki_path: str) -> Tuple[Wiki, Dict[str, str]]:
    """
    Read results/compiled_wiki.jsonl.
    Returns:
      wiki: Wiki object
      sources: {source_id: source_text}
    """
    claims = []
    sources: Dict[str, str] = {}

    with open(wiki_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Support two record types: claim records and source records
            if rec.get("type") == "source" or ("source_id" in rec and "text" in rec
                                                and "claim_id" not in rec):
                sources[rec["source_id"]] = rec["text"]
            else:
                claim_id = rec.get("claim_id", rec.get("id", f"c{len(claims)}"))
                text = rec.get("text", rec.get("claim", ""))
                source_ids = rec.get("source_ids", rec.get("sources", []))
                fusion_type = rec.get("fusion_type", "unknown")
                sole_entailer = rec.get("sole_entailer", None)
                if text:
                    claims.append(Claim(
                        claim_id=str(claim_id),
                        text=text,
                        source_ids=[str(s) for s in source_ids],
                        fusion_type=fusion_type,
                        sole_entailer=str(sole_entailer) if sole_entailer else None,
                    ))

    print(f"[load] {len(claims)} claims, {len(sources)} sources", flush=True)
    return Wiki(claims), sources


def load_corpus_dir(corpus_dir: str) -> Dict[str, str]:
    """
    Load Wikipedia corpus from directory of .txt or .jsonl files.
    Returns {doc_id: text}.
    """
    corpus: Dict[str, str] = {}
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        print(f"[warn] corpus dir not found: {corpus_dir}", flush=True)
        return corpus

    for fp in sorted(corpus_path.iterdir()):
        if fp.suffix == ".jsonl":
            with open(fp) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    doc_id = str(rec.get("id", rec.get("doc_id", fp.stem)))
                    text = rec.get("text", rec.get("content", ""))
                    if text:
                        corpus[doc_id] = text
        elif fp.suffix == ".txt":
            corpus[fp.stem] = fp.read_text(errors="replace")

    print(f"[load] corpus: {len(corpus)} docs from {corpus_dir}", flush=True)
    return corpus


# ---------------------------------------------------------------------------
# Step 2: Select K=20 retraction candidates
# ---------------------------------------------------------------------------

def build_cocitation_graph(wiki: Wiki) -> Dict[str, int]:
    """
    Build BM25 co-citation graph; return {source_id: co-citation_degree}.
    Co-citation degree = number of other sources that appear in the same claim.
    """
    degree: Dict[str, int] = defaultdict(int)
    for claim in wiki.claims:
        sids = claim.source_ids
        if len(sids) > 1:
            for sid in sids:
                degree[sid] += len(sids) - 1
        else:
            for sid in sids:
                if sid not in degree:
                    degree[sid] = 0
    return dict(degree)


def count_multi_source_appearances(wiki: Wiki) -> Dict[str, int]:
    """Count how many multi_source_only claims each source appears in."""
    counts: Dict[str, int] = defaultdict(int)
    for claim in wiki.claims:
        if claim.fusion_type == "multi_source_only" and len(claim.source_ids) >= 3:
            for sid in claim.source_ids:
                counts[sid] += 1
    return dict(counts)


def count_fused_with_cosources(wiki: Wiki) -> Dict[str, int]:
    """Count how many fused claims (>=3 co-sources) each source contributes to."""
    counts: Dict[str, int] = defaultdict(int)
    for claim in wiki.claims:
        if len(claim.source_ids) >= 3:
            for sid in claim.source_ids:
                counts[sid] += 1
    return dict(counts)


def count_claims_per_source(wiki: Wiki) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for claim in wiki.claims:
        for sid in claim.source_ids:
            counts[sid] += 1
    return dict(counts)


def detect_numeric_claims(wiki: Wiki) -> Dict[str, List[str]]:
    """Map source_id -> list of claims with unique numeric content."""
    numeric_re = re.compile(r'\b\d+[\d,\.]*\b')
    result: Dict[str, List[str]] = defaultdict(list)
    for claim in wiki.claims:
        nums = numeric_re.findall(claim.text)
        if nums and len(claim.source_ids) == 1:
            result[claim.source_ids[0]].append(claim.claim_id)
    return dict(result)


def select_candidates(wiki: Wiki, seed: int = 42, n_target: int = 20,
                       force_include: Optional[List[str]] = None) -> List[str]:
    """
    Select n_target retraction candidates stratified by design.
    Bucket ratios (4:4:4:4:2:2 = 20) are scaled to n_target so the n=20 design is
    preserved exactly at n_target=20 and a superset is produced for larger n.
    force_include (e.g. the already-computed candidates) are pinned first so their
    cached oracle recompiles are reused.
    Returns list of source_ids.
    """
    rng = random.Random(seed)
    degree = build_cocitation_graph(wiki)
    claims_per_source = count_claims_per_source(wiki)
    multi_heavy = count_multi_source_appearances(wiki)
    fused_heavy = count_fused_with_cosources(wiki)
    numeric_sources = detect_numeric_claims(wiki)

    all_sources = list(degree.keys())
    if not all_sources:
        raise ValueError("No sources found in compiled wiki — check wiki_path")

    sorted_by_degree = sorted(all_sources, key=lambda s: degree.get(s, 0), reverse=True)

    selected = []
    used = set()

    # Pin previously-computed candidates first (cache reuse on resume).
    if force_include:
        for s in force_include:
            if s in degree and s not in used:
                selected.append(s); used.add(s)

    def pick_n(candidates: List[str], n: int) -> List[str]:
        out = []
        for s in candidates:
            if s not in used and len(out) < n:
                out.append(s)
                used.add(s)
        # If not enough, fill randomly
        remaining = [s for s in all_sources if s not in used]
        rng.shuffle(remaining)
        while len(out) < n and remaining:
            s = remaining.pop()
            out.append(s)
            used.add(s)
        return out

    # Scale bucket sizes to n_target (ratios 0.2/0.2/0.2/0.2/0.1/0.1).
    nb = max(1, round(0.20 * n_target))
    ne = max(1, round(0.10 * n_target))
    bucket_sizes = {"hubs": nb, "sing": nb, "fus": nb, "rand": nb, "easy": ne}
    # hard bucket absorbs rounding remainder to land exactly on n_target
    n_hard = max(1, n_target - (nb * 4 + ne))

    # hub sources (highest co-citation degree)
    hubs = pick_n(sorted_by_degree, bucket_sizes["hubs"])
    selected += hubs

    # singleton sources (lowest degree, ≤2 claims)
    singletons_candidates = sorted(
        [s for s in all_sources if claims_per_source.get(s, 0) <= 2],
        key=lambda s: degree.get(s, 0)
    )
    if len(singletons_candidates) < bucket_sizes["sing"]:
        singletons_candidates = sorted(all_sources, key=lambda s: degree.get(s, 0))
    singletons = pick_n(singletons_candidates, bucket_sizes["sing"])
    selected += singletons

    # fusion-heavy (appears in >=3 multi_source_only claims)
    fusion_candidates = sorted(
        [s for s in all_sources if multi_heavy.get(s, 0) >= 3],
        key=lambda s: multi_heavy.get(s, 0), reverse=True
    )
    if len(fusion_candidates) < bucket_sizes["fus"]:
        fusion_candidates = sorted(all_sources, key=lambda s: multi_heavy.get(s, 0), reverse=True)
    fusion_heavy_list = pick_n(fusion_candidates, bucket_sizes["fus"])
    selected += fusion_heavy_list

    # random sources
    random_pool = [s for s in all_sources if s not in used]
    rng.shuffle(random_pool)
    random_picks = pick_n(random_pool, bucket_sizes["rand"])
    selected += random_picks

    # easy controls (unique numeric claims)
    numeric_candidates = sorted(
        [s for s in numeric_sources if s not in used],
        key=lambda s: len(numeric_sources.get(s, [])), reverse=True
    )
    if len(numeric_candidates) < bucket_sizes["easy"]:
        numeric_candidates = [s for s in all_sources if s not in used]
    easy_controls = pick_n(numeric_candidates, bucket_sizes["easy"])
    selected += easy_controls

    # hard cases (contributes to 5+ fused claims with >=3 co-sources)
    hard_candidates = sorted(
        [s for s in all_sources if fused_heavy.get(s, 0) >= 5],
        key=lambda s: fused_heavy.get(s, 0), reverse=True
    )
    if len(hard_candidates) < n_hard:
        hard_candidates = sorted(all_sources, key=lambda s: fused_heavy.get(s, 0), reverse=True)
    hard_cases = pick_n(hard_candidates, n_hard)
    selected += hard_cases

    print(f"[candidates] selected {len(selected)} sources:", flush=True)
    print(f"  hubs={hubs}", flush=True)
    print(f"  singletons={singletons}", flush=True)
    print(f"  fusion_heavy={fusion_heavy_list}", flush=True)
    print(f"  random={random_picks}", flush=True)
    print(f"  easy_controls={easy_controls}", flush=True)
    print(f"  hard_cases={hard_cases}", flush=True)

    return selected


def save_candidates(candidates: List[str], wiki: Wiki, out_path: str):
    degree = build_cocitation_graph(wiki)
    claims_per = count_claims_per_source(wiki)
    multi_heavy = count_multi_source_appearances(wiki)
    fused_heavy = count_fused_with_cosources(wiki)

    records = []
    for i, sid in enumerate(candidates):
        records.append({
            "rank": i,
            "source_id": sid,
            "cocitation_degree": degree.get(sid, 0),
            "n_claims": claims_per.get(sid, 0),
            "multi_source_appearances": multi_heavy.get(sid, 0),
            "fused_claim_count": fused_heavy.get(sid, 0),
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(records, fh, indent=2)
    print(f"[candidates] saved to {out_path}", flush=True)


# ---------------------------------------------------------------------------
# LLM client (vLLM HTTP)
# ---------------------------------------------------------------------------

class VLLMClient:
    """Thin HTTP client for vLLM OpenAI-compatible endpoint."""

    def __init__(self, host: str = "http://127.0.0.1:8102",
                 model: str = "Qwen/Qwen3-14B"):
        self.host = host.rstrip("/")
        self.model = model
        self.n_calls = 0

    def _post(self, messages: List[dict], max_tokens: int = 512,
              temperature: float = 0.0) -> str:
        import urllib.request
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        self.n_calls += 1
        return data["choices"][0]["message"]["content"].strip()

    @staticmethod
    def _strip_thinking(text: str) -> str:
        import re
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def gen(self, system: str, user: str, max_tokens: int = 512,
            temperature: float = 0.0) -> str:
        messages = []
        # /no_think disables Qwen3 chain-of-thought mode via system message suffix
        sys_content = (system + "\n/no_think") if system else "/no_think"
        messages.append({"role": "system", "content": sys_content})
        messages.append({"role": "user", "content": user})
        raw = self._post(messages, max_tokens=max_tokens, temperature=temperature)
        return self._strip_thinking(raw)

    def rewrite_without_source(self, original_claim: str,
                                removed_source_text: str,
                                surviving_evidence: List[str]) -> str:
        surviving_joined = "\n\n".join(surviving_evidence)[:1000]
        removed_snip = removed_source_text[:500]
        prompt = (
            "The following claim was partially supported by source [REMOVED_SOURCE]. "
            "That source is being retracted. Rewrite the claim using ONLY the surviving "
            "evidence provided, preserving everything that is still supported. "
            "If nothing is supported by surviving evidence, output only: [DELETED].\n\n"
            f"ORIGINAL CLAIM: {original_claim}\n\n"
            f"REMOVED SOURCE (retracted): {removed_snip}\n\n"
            f"SURVIVING EVIDENCE: {surviving_joined}\n\n"
            "REWRITTEN CLAIM:"
        )
        return self.gen("", prompt, max_tokens=256, temperature=0.0)

    def compile_wiki(self, docs: Dict[str, str], seed: int = 0) -> List[dict]:
        """
        Recompile a wiki from a corpus dict {doc_id: text}.
        Returns list of claim dicts with keys: text, source_ids, fusion_type, sole_entailer.
        """
        # Group docs into pages of ~8 docs each
        doc_ids = list(docs.keys())
        random.Random(seed).shuffle(doc_ids)
        page_size = 8
        pages = [doc_ids[i:i + page_size] for i in range(0, len(doc_ids), page_size)]

        compile_sys = (
            "You are a knowledge compiler. Given a set of source documents, extract "
            "a list of atomic factual claims. For each claim, output a JSON object on "
            "its own line with fields: "
            "'text' (the claim), "
            "'source_ids' (list of doc IDs that support this claim), "
            "'fusion_type' ('single_source' if only one source, 'multi_source_only' if fused), "
            "'sole_entailer' (the single doc ID if fusion_type is single_source, else null). "
            "Output ONLY JSON lines, no other text."
        )

        all_claims = []
        for pi, page in enumerate(pages):
            doc_block = "\n\n".join(
                f"[DOC {did}]\n{docs[did][:800]}" for did in page
            )
            user_msg = f"Extract claims from these documents:\n\n{doc_block}"
            try:
                resp = self.gen(compile_sys, user_msg, max_tokens=2048, temperature=0.3)
                for line in resp.splitlines():
                    line = line.strip()
                    if not line or not line.startswith("{"):
                        continue
                    try:
                        rec = json.loads(line)
                        if "text" in rec:
                            all_claims.append(rec)
                    except json.JSONDecodeError:
                        pass
            except Exception as e:
                print(f"  [compile] page {pi} error: {e}", flush=True)

        return all_claims


# ---------------------------------------------------------------------------
# Step 3: Build oracle wikis
# ---------------------------------------------------------------------------

def cosine_sim_matrix(embs_a: np.ndarray, embs_b: np.ndarray) -> np.ndarray:
    """Return (len_a, len_b) cosine similarity matrix."""
    norms_a = np.linalg.norm(embs_a, axis=1, keepdims=True) + 1e-9
    norms_b = np.linalg.norm(embs_b, axis=1, keepdims=True) + 1e-9
    return (embs_a / norms_a) @ (embs_b / norms_b).T


def label_claims_vs_oracle(
    original_claims: List[Claim],
    oracle_claim_texts: List[str],
    mode: str = "cpu",
) -> Dict[str, str]:
    """
    Label each original claim as DIES / CHANGES / SURVIVES against oracle.
    Uses BGE-small cosine similarity, with NLI disambiguation in the CHANGES zone.
    Returns {claim_id: label}.
    """
    bge = _get_bge()
    labels: Dict[str, str] = {}

    if not oracle_claim_texts:
        # No oracle claims at all → everything dies
        for claim in original_claims:
            labels[claim.claim_id] = "DIES"
        return labels

    orig_texts = [c.text for c in original_claims]
    orig_embs = bge.encode(orig_texts, normalize_embeddings=True, show_progress_bar=False)
    oracle_embs = bge.encode(oracle_claim_texts, normalize_embeddings=True, show_progress_bar=False)

    sim_matrix = cosine_sim_matrix(orig_embs, oracle_embs)  # (n_orig, n_oracle)
    max_sims = sim_matrix.max(axis=1)                       # (n_orig,)
    best_matches = sim_matrix.argmax(axis=1)                # (n_orig,)

    changes_zone_pairs = []   # (orig_idx, oracle_idx)
    for i, claim in enumerate(original_claims):
        ms = float(max_sims[i])
        if ms < SIM_DIE_THRESHOLD:
            labels[claim.claim_id] = "DIES"
        elif ms >= SIM_SURVIVE_THRESHOLD:
            labels[claim.claim_id] = "SURVIVES"
        else:
            labels[claim.claim_id] = "CHANGES"  # tentative; disambiguate with NLI
            changes_zone_pairs.append((i, int(best_matches[i])))

    # NLI disambiguation for CHANGES zone
    if changes_zone_pairs:
        nli = _get_nli(mode)
        for orig_idx, oracle_idx in changes_zone_pairs:
            orig_text = original_claims[orig_idx].text
            oracle_text = oracle_claim_texts[oracle_idx]
            try:
                result = nli(f"{orig_text} [SEP] {oracle_text}")
                top = max(result, key=lambda x: x["score"])
                if top["label"].upper() == "ENTAILMENT":
                    labels[original_claims[orig_idx].claim_id] = "SURVIVES"
                elif top["label"].upper() == "CONTRADICTION":
                    labels[original_claims[orig_idx].claim_id] = "DIES"
                # else keep CHANGES
            except Exception as e:
                print(f"  [nli] error for claim {original_claims[orig_idx].claim_id}: {e}",
                      flush=True)

    return labels


def build_oracle_wiki(
    sk: str,
    wiki: Wiki,
    sources: Dict[str, str],
    corpus_dir: str,
    llm: Optional[VLLMClient],
    oracle_dir: str,
    mode: str = "cpu",
) -> Dict[str, str]:
    """
    Recompile wiki without Sk; label original claims as DIES/CHANGES/SURVIVES.
    Returns {claim_id: label}.
    Saves oracle diff to oracle_dir/{sk}_oracle.jsonl.
    """
    os.makedirs(oracle_dir, exist_ok=True)
    oracle_path = os.path.join(oracle_dir, f"{sk}_oracle.jsonl")

    # If already computed, load from disk
    if os.path.exists(oracle_path):
        print(f"  [oracle] loading cached oracle for {sk}", flush=True)
        labels = {}
        with open(oracle_path) as fh:
            for line in fh:
                rec = json.loads(line.strip())
                if "claim_id" in rec and "oracle_label" in rec:
                    labels[rec["claim_id"]] = rec["oracle_label"]
        if labels:
            return labels

    print(f"  [oracle] building oracle for {sk} ...", flush=True)
    t0 = time.time()

    # Claims that reference Sk at all
    sk_claims = [c for c in wiki.claims if sk in c.source_ids]
    non_sk_claims = [c for c in wiki.claims if sk not in c.source_ids]

    # If no LLM, use heuristic oracle: claims that are single_source from Sk → DIES,
    # multi-source with only Sk as entailer → DIES, multi-source with others → SURVIVES
    if llm is None:
        labels = {}
        for claim in non_sk_claims:
            labels[claim.claim_id] = "SURVIVES"
        for claim in sk_claims:
            remaining = [s for s in claim.source_ids if s != sk]
            if not remaining:
                labels[claim.claim_id] = "DIES"
            elif claim.fusion_type == "single_source" and claim.sole_entailer == sk:
                labels[claim.claim_id] = "DIES"
            else:
                labels[claim.claim_id] = "CHANGES"  # conservative
        # Save
        with open(oracle_path, "w") as fh:
            for claim in wiki.claims:
                rec = claim.to_dict()
                rec["oracle_label"] = labels.get(claim.claim_id, "SURVIVES")
                fh.write(json.dumps(rec) + "\n")
        return labels

    # Full oracle: recompile corpus \ {Sk}
    # Build corpus_minus_k from sources dict (or corpus dir as fallback)
    corpus_minus_k: Dict[str, str] = {}
    if sources:
        corpus_minus_k = {sid: text for sid, text in sources.items() if sid != sk}
    else:
        corpus_all = load_corpus_dir(corpus_dir)
        corpus_minus_k = {did: text for did, text in corpus_all.items() if did != sk}

    if not corpus_minus_k:
        print(f"  [oracle] WARNING: empty corpus_minus_k for {sk}, using heuristics", flush=True)
        return build_oracle_wiki(sk, wiki, sources, corpus_dir, None, oracle_dir, mode)

    oracle_claims_raw = llm.compile_wiki(corpus_minus_k, seed=42)
    oracle_texts = [r["text"] for r in oracle_claims_raw if r.get("text")]

    # Label original claims against oracle
    labels = label_claims_vs_oracle(wiki.claims, oracle_texts, mode=mode)

    # Save oracle diff
    with open(oracle_path, "w") as fh:
        for claim in wiki.claims:
            rec = claim.to_dict()
            rec["oracle_label"] = labels.get(claim.claim_id, "SURVIVES")
            fh.write(json.dumps(rec) + "\n")
        # Also save oracle claims
        for oc in oracle_claims_raw:
            fh.write(json.dumps({"type": "oracle_claim", **oc}) + "\n")

    elapsed = time.time() - t0
    n_dies = sum(1 for v in labels.values() if v == "DIES")
    n_changes = sum(1 for v in labels.values() if v == "CHANGES")
    n_survives = sum(1 for v in labels.values() if v == "SURVIVES")
    print(f"  [oracle] {sk}: DIES={n_dies} CHANGES={n_changes} SURVIVES={n_survives} "
          f"({elapsed:.1f}s)", flush=True)

    return labels


# ---------------------------------------------------------------------------
# Step 4: partial_delete (our method)
# ---------------------------------------------------------------------------

def get_source_text(sid: str, sources: Dict[str, str]) -> str:
    return sources.get(sid, f"[source {sid} not found]")


def partial_delete(wiki: Wiki, sk: str, sources: Dict[str, str],
                   llm: Optional[VLLMClient],
                   rewrite_threshold: int = 3) -> Wiki:
    """
    Run partial_delete algorithm to remove source Sk from wiki.
    Returns a new Wiki with claims updated/deleted accordingly.

    rewrite_threshold: if a claim has >= this many sources remaining after
    removing Sk, the claim SURVIVES (text unchanged; Sk just dropped from
    source list).  Only LLM-rewrite when Sk was a primary contributor
    (few remaining sources).  This prevents over-editing when Sk is one of
    many co-sources in a broad-coverage claim.
    """
    new_wiki = copy.deepcopy(wiki)
    to_remove = []

    for claim in new_wiki.claims:
        if sk not in claim.source_ids:
            continue  # claim doesn't cite Sk → SURVIVES unchanged
        remaining = [s for s in claim.source_ids if s != sk]

        # Case 1: sole source → DIES
        if (not remaining) or (
            claim.fusion_type == "single_source" and claim.sole_entailer == sk
        ):
            to_remove.append(claim)

        # Case 2: Sk was marginal (many co-sources remain) → SURVIVES, just update provenance
        elif len(remaining) >= rewrite_threshold:
            claim.source_ids = remaining  # provenance update only, no text change

        # Case 3: Sk was a primary contributor (few remaining) → LLM rewrite
        else:
            surviving_evidence = [get_source_text(s, sources) for s in remaining]
            if llm is not None:
                rewritten = llm.rewrite_without_source(
                    original_claim=claim.text,
                    removed_source_text=get_source_text(sk, sources),
                    surviving_evidence=surviving_evidence,
                )
                if rewritten.strip() == "[DELETED]":
                    to_remove.append(claim)
                else:
                    claim.text = rewritten
                    claim.source_ids = remaining
            else:
                claim.source_ids = remaining

    for claim in to_remove:
        new_wiki.remove_claim(claim)

    return new_wiki


def predict_labels_partial_delete(
    original_wiki: Wiki,
    updated_wiki: Wiki,
    mode: str = "cpu",
) -> Dict[str, str]:
    """
    Given original and updated wiki, compute predicted DIES/CHANGES/SURVIVES labels.
    """
    orig_ids = {c.claim_id: c for c in original_wiki.claims}
    updated_ids = {c.claim_id: c for c in updated_wiki.claims}

    # Claims not present in updated → DIES
    # Claims present with same text → SURVIVES
    # Claims present with different text → CHANGES
    labels: Dict[str, str] = {}
    for cid, claim in orig_ids.items():
        if cid not in updated_ids:
            labels[cid] = "DIES"
        elif updated_ids[cid].text.strip() == claim.text.strip():
            labels[cid] = "SURVIVES"
        else:
            labels[cid] = "CHANGES"

    return labels


# ---------------------------------------------------------------------------
# Step 5: Baselines
# ---------------------------------------------------------------------------

def run_baseline_rag_drop(
    sk: str,
    wiki: Wiki,
    sources: Dict[str, str],
    mode: str = "cpu",
) -> Dict[str, str]:
    """
    Baseline 1 (RAG-drop):
    Remove all BM25 chunks from Sk; re-retrieve for each original claim.
    Classify retrieved answer as DIES/CHANGES/SURVIVES vs oracle.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("  [baseline] rank_bm25 not installed, skipping RAG-drop", flush=True)
        return {}

    # Build BM25 index over remaining sources
    remaining_sources = {sid: text for sid, text in sources.items() if sid != sk}
    if not remaining_sources:
        return {c.claim_id: "DIES" for c in wiki.claims}

    source_ids_list = list(remaining_sources.keys())
    tokenized_corpus = [remaining_sources[sid].lower().split() for sid in source_ids_list]
    bm25 = BM25Okapi(tokenized_corpus)

    labels: Dict[str, str] = {}
    for claim in wiki.claims:
        if sk not in claim.source_ids:
            labels[claim.claim_id] = "SURVIVES"
            continue
        # Check if the claim can be re-retrieved without Sk
        query_tokens = claim.text.lower().split()
        scores = bm25.get_scores(query_tokens)
        top_idx = int(np.argmax(scores))
        top_score = float(scores[top_idx])

        if top_score < 0.1:
            labels[claim.claim_id] = "DIES"
        else:
            # Check if any original source (other than sk) is in top results
            top_k = np.argsort(scores)[::-1][:5]
            top_sources = {source_ids_list[i] for i in top_k}
            original_remaining = set(claim.source_ids) - {sk}
            if top_sources & original_remaining:
                labels[claim.claim_id] = "SURVIVES"
            else:
                labels[claim.claim_id] = "CHANGES"

    return labels


def _compute_tfidf_ngrams(text: str, n: int = 2) -> Dict[str, float]:
    """Simple TF-IDF n-gram computation (single document)."""
    tokens = text.lower().split()
    ngrams = [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return {}
    freq: Dict[str, int] = defaultdict(int)
    for ng in ngrams:
        freq[ng] += 1
    total = len(ngrams)
    return {ng: count / total for ng, count in freq.items()}


def run_baseline_naive_delete(
    sk: str,
    wiki: Wiki,
    sources: Dict[str, str],
) -> Dict[str, str]:
    """
    Baseline 2 (Naive-wiki-delete):
    Delete all wiki claims containing unique Sk n-grams (tf-idf > 0.8).
    """
    sk_text = sources.get(sk, "")
    sk_ngrams = _compute_tfidf_ngrams(sk_text, n=2)
    unique_sk_ngrams = {ng for ng, score in sk_ngrams.items() if score > 0.8}

    labels: Dict[str, str] = {}
    for claim in wiki.claims:
        if sk not in claim.source_ids:
            labels[claim.claim_id] = "SURVIVES"
            continue
        claim_lower = claim.text.lower()
        # Check if claim contains any unique Sk n-grams
        hit = any(ng in claim_lower for ng in unique_sk_ngrams)
        if hit:
            labels[claim.claim_id] = "DIES"
        else:
            labels[claim.claim_id] = "SURVIVES"
        # CHANGES = 0 by design (structural)

    return labels


def run_baseline_flat(
    sk: str,
    wiki: Wiki,
    sources: Dict[str, str],
    gpu_id: int,
    flat_dir: str,
) -> Optional[Dict[str, str]]:
    """
    Baseline 3 (FLAT/param-unlearn):
    Apply FLAT fine-tuning on forget set = Sk text.
    Probe with paraphrase queries for each Sk claim.
    Returns None if GPU not available.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            print("  [flat] CUDA not available, skipping FLAT baseline", flush=True)
            return None
    except ImportError:
        print("  [flat] torch not installed, skipping FLAT baseline", flush=True)
        return None

    flat_script = os.path.join(flat_dir, "unlearn.py")
    if not os.path.exists(flat_script):
        print(f"  [flat] FLAT script not found at {flat_script}, skipping", flush=True)
        return None

    # In production: run FLAT unlearning and probe
    # For now, mark as N/A (requires FLAT-specific setup)
    print(f"  [flat] FLAT baseline requires FLAT-specific setup (model finetuning). "
          f"Marking as N/A for source {sk}.", flush=True)
    return None


# ---------------------------------------------------------------------------
# Step 6: Compute metrics
# ---------------------------------------------------------------------------

def compute_oracle_f1(predicted: Dict[str, str],
                      oracle: Dict[str, str]) -> Dict[str, float]:
    """
    Compute macro F1 between predicted and oracle labels.
    Labels: DIES, CHANGES, SURVIVES.
    Returns dict with oracle_f1, residue_rate, over_deletion_rate, change_acc.
    """
    from sklearn.metrics import f1_score, confusion_matrix

    common_ids = [cid for cid in oracle if cid in predicted]
    if not common_ids:
        return {
            "oracle_f1": 0.0,
            "residue_rate": 0.0,
            "over_deletion_rate": 0.0,
            "change_acc": 0.0,
            "n": 0,
        }

    label_map = {"DIES": 0, "CHANGES": 1, "SURVIVES": 2}
    y_true = [label_map[oracle[cid]] for cid in common_ids]
    y_pred = [label_map.get(predicted.get(cid, "SURVIVES"), 2) for cid in common_ids]

    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Confusion matrix: rows=true, cols=pred
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    # cm[i,j] = true label i, predicted label j

    # Residue-rate: P(predicted=SURVIVES | oracle=DIES)
    n_oracle_dies = cm[0, :].sum()
    residue_rate = float(cm[0, 2]) / max(n_oracle_dies, 1)

    # Over-deletion-rate: P(predicted=DIES | oracle=SURVIVES)
    n_oracle_survives = cm[2, :].sum()
    over_deletion_rate = float(cm[2, 0]) / max(n_oracle_survives, 1)

    # ChangeAcc: P(predicted=CHANGES | oracle=CHANGES)
    n_oracle_changes = cm[1, :].sum()
    change_acc = float(cm[1, 1]) / max(n_oracle_changes, 1)

    return {
        "oracle_f1": round(float(f1), 4),
        "residue_rate": round(residue_rate, 4),
        "over_deletion_rate": round(over_deletion_rate, 4),
        "change_acc": round(change_acc, 4),
        "n": len(common_ids),
    }


def compute_stratum_metrics(
    predicted: Dict[str, str],
    oracle: Dict[str, str],
    wiki: Wiki,
    sk: str,
) -> Dict[str, Dict]:
    """
    Compute metrics stratified by Fused-F (multi_source_only) and Single-S (single_source).
    """
    fused_ids = {c.claim_id for c in wiki.claims
                 if sk in c.source_ids and c.fusion_type == "multi_source_only"}
    single_ids = {c.claim_id for c in wiki.claims
                  if sk in c.source_ids and c.fusion_type == "single_source"}

    pred_fused = {cid: predicted[cid] for cid in fused_ids if cid in predicted}
    oracle_fused = {cid: oracle[cid] for cid in fused_ids if cid in oracle}
    pred_single = {cid: predicted[cid] for cid in single_ids if cid in predicted}
    oracle_single = {cid: oracle[cid] for cid in single_ids if cid in oracle}
    pred_all = {cid: predicted[cid] for cid in oracle if cid in predicted}

    return {
        "all": compute_oracle_f1(pred_all, oracle),
        "fused_F": compute_oracle_f1(pred_fused, oracle_fused),
        "single_S": compute_oracle_f1(pred_single, oracle_single),
    }


def bootstrap_ci(values: List[float], n_iter: int = BOOTSTRAP_ITERS,
                 seed: int = 0) -> Tuple[float, float]:
    """Return 95% bootstrap CI for the mean of values."""
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_iter)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def wilcoxon_test(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Wilcoxon signed-rank test; returns (W, p_value)."""
    if len(a) < 2 or len(b) < 2 or len(a) != len(b):
        return (float("nan"), float("nan"))
    try:
        from scipy.stats import wilcoxon
        result = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return (float(result.statistic), float(result.pvalue))
    except Exception as e:
        print(f"  [stats] wilcoxon error: {e}", flush=True)
        return (float("nan"), float("nan"))


def compute_statistical_tests(
    our_f1s: List[float],
    baseline_f1s_dict: Dict[str, List[float]],
) -> Dict[str, dict]:
    """
    Wilcoxon signed-rank tests and bootstrap CIs.
    our_f1s: per-source Oracle-F1 values for our method.
    baseline_f1s_dict: {baseline_name: [per-source F1 values]}.
    Returns stats dict.
    """
    stats = {}
    our_ci = bootstrap_ci(our_f1s)
    stats["our_method"] = {
        "mean_f1": round(float(np.mean(our_f1s)), 4) if our_f1s else 0.0,
        "ci_95": [round(our_ci[0], 4), round(our_ci[1], 4)],
    }

    for name, bline_f1s in baseline_f1s_dict.items():
        if not bline_f1s or all(v != v for v in bline_f1s):  # NaN check
            stats[name] = {"mean_f1": "N/A", "ci_95": "N/A", "W": "N/A", "p": "N/A"}
            continue
        # Align lengths
        min_len = min(len(our_f1s), len(bline_f1s))
        W, p = wilcoxon_test(our_f1s[:min_len], bline_f1s[:min_len])
        bline_ci = bootstrap_ci(bline_f1s)
        stats[name] = {
            "mean_f1": round(float(np.mean(bline_f1s)), 4),
            "ci_95": [round(bline_ci[0], 4), round(bline_ci[1], 4)],
            "W": round(W, 2) if W == W else "N/A",
            "p": round(p, 4) if p == p else "N/A",
            "significant": bool(p < BONFERRONI_ALPHA) if p == p else False,
        }

    return stats


# ---------------------------------------------------------------------------
# Step 7: Save results
# ---------------------------------------------------------------------------

def format_table(results: dict) -> str:
    """Format results as markdown table matching paper format."""
    lines = [
        "| Source | Method | Stratum | Oracle-F1 | Residue-Rate | Over-Del-Rate | ChangeAcc |",
        "|--------|--------|---------|-----------|--------------|---------------|-----------|",
    ]
    for sk, src_results in sorted(results.items()):
        for method_name, method_data in sorted(src_results.items()):
            if not isinstance(method_data, dict) or "all" not in method_data:
                continue
            for stratum in ["all", "fused_F", "single_S"]:
                m = method_data.get(stratum, {})
                if not m:
                    continue
                lines.append(
                    f"| {sk[:12]} | {method_name} | {stratum} "
                    f"| {m.get('oracle_f1', 'N/A')} "
                    f"| {m.get('residue_rate', 'N/A')} "
                    f"| {m.get('over_deletion_rate', 'N/A')} "
                    f"| {m.get('change_acc', 'N/A')} |"
                )
    return "\n".join(lines)


def format_report(results: dict, stats: dict) -> str:
    """Format human-readable interpretation report."""
    lines = [
        "# E1 Oracle-Matched Partial Deletion — Results Report",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
    ]

    our_f1s = []
    baseline_data = defaultdict(list)
    for sk, src_results in results.items():
        our = src_results.get("partial_delete", {}).get("all", {})
        if our:
            our_f1s.append(our.get("oracle_f1", 0.0))
        for bname in ["rag_drop", "naive_delete", "flat"]:
            bm = src_results.get(bname, {}).get("all", {})
            if bm and bm.get("oracle_f1") is not None:
                baseline_data[bname].append(bm.get("oracle_f1", 0.0))

    mean_our = float(np.mean(our_f1s)) if our_f1s else 0.0
    lines.append(f"Our method (partial_delete) mean Oracle-F1: {mean_our:.4f}")
    for bname, bvals in baseline_data.items():
        lines.append(f"Baseline {bname} mean Oracle-F1: {float(np.mean(bvals)):.4f}")

    lines += [
        "",
        "## Statistical Tests (Wilcoxon signed-rank, Bonferroni α_eff=0.017)",
        "",
    ]
    for name, s in stats.items():
        if name == "our_method":
            lines.append(f"Our method: mean={s['mean_f1']}, CI={s['ci_95']}")
        else:
            sig = s.get("significant", False)
            lines.append(
                f"{name}: mean={s['mean_f1']}, CI={s['ci_95']}, "
                f"W={s.get('W', 'N/A')}, p={s.get('p', 'N/A')}, "
                f"significant={sig}"
            )

    lines += [
        "",
        "## Decision",
        "",
        "C1 CONFIRMED if our method Oracle-F1 significantly exceeds all baselines "
        "(p < 0.017) with mean F1 >= 0.70.",
        "",
    ]
    if mean_our >= 0.70:
        lines.append(f"RESULT: mean Oracle-F1 = {mean_our:.4f} >= 0.70 threshold.")
    else:
        lines.append(f"RESULT: mean Oracle-F1 = {mean_our:.4f} < 0.70 threshold. "
                     f"Review edge cases.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="E1: Oracle-Matched Partial Deletion")
    ap.add_argument("--mode", choices=["cpu", "gpu"], default="cpu",
                    help="cpu: skip GPU experiments (FLAT), use DeBERTa for NLI; "
                         "gpu: full GPU run with Qwen3-14B")
    ap.add_argument("--gpu-id", type=int, default=4,
                    help="GPU device ID for full GPU run")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8102")),
                    help="vLLM server port")
    ap.add_argument("--wiki-path", default=DEFAULT_WIKI_PATH,
                    help="Path to compiled_wiki.jsonl from E0")
    ap.add_argument("--corpus-dir", default=DEFAULT_CORPUS_DIR,
                    help="Directory containing Wikipedia corpus")
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                    help="Output results directory")
    ap.add_argument("--oracle-dir", default=DEFAULT_ORACLE_DIR,
                    help="Output oracle_diffs directory")
    ap.add_argument("--flat-dir",
                    default=os.path.join(EXPERIMENT_ROOT, "FLAT"),
                    help="Path to FLAT unlearning directory")
    ap.add_argument("--k-candidates", type=int, default=K_CANDIDATES,
                    help="Number of retraction candidates to select")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4,
                    help="Number of parallel workers for oracle building")
    ap.add_argument("--skip-oracle", action="store_true",
                    help="Skip oracle recompilation (use cached oracle_diffs)")
    ap.add_argument("--skip-flat", action="store_true",
                    help="Always skip FLAT baseline")
    ap.add_argument("--model", default="Qwen/Qwen3-14B",
                    help="vLLM model name")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.oracle_dir, exist_ok=True)

    if args.mode == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    t_start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Load compiled wiki
    # ------------------------------------------------------------------
    print("=== Step 1: Load compiled wiki ===", flush=True)
    if not os.path.exists(args.wiki_path):
        print(f"[WARN] wiki not found at {args.wiki_path}", flush=True)
        print("Creating a minimal stub wiki for testing pipeline integrity ...", flush=True)
        os.makedirs(os.path.dirname(args.wiki_path), exist_ok=True)
        stub_claims = [
            {"claim_id": f"c{i}", "text": f"Stub claim {i} from source s{i % 5}.",
             "source_ids": [f"s{i % 5}"],
             "fusion_type": "single_source",
             "sole_entailer": f"s{i % 5}"}
            for i in range(30)
        ] + [
            {"claim_id": f"cf{i}", "text": f"Fused claim {i} from sources s0 and s1.",
             "source_ids": ["s0", "s1", f"s{i + 2}"],
             "fusion_type": "multi_source_only",
             "sole_entailer": None}
            for i in range(10)
        ]
        stub_sources = [
            {"type": "source", "source_id": f"s{i}",
             "text": f"Source text for document s{i}. Contains facts about topic {i}."}
            for i in range(10)
        ]
        with open(args.wiki_path, "w") as fh:
            for rec in stub_claims + stub_sources:
                fh.write(json.dumps(rec) + "\n")
        print(f"[stub] wrote {len(stub_claims)} claims + {len(stub_sources)} sources "
              f"to {args.wiki_path}", flush=True)

    wiki, sources = load_compiled_wiki(args.wiki_path)

    # ------------------------------------------------------------------
    # Step 2: Select candidates
    # ------------------------------------------------------------------
    print("\n=== Step 2: Select retraction candidates ===", flush=True)
    candidates_path = os.path.join(args.results_dir, "e1_retraction_candidates.json")
    existing = []
    if os.path.exists(candidates_path):
        with open(candidates_path) as fh:
            existing = [r["source_id"] for r in json.load(fh)]
    if existing and len(existing) >= args.k_candidates:
        # already have enough; reuse as-is
        print(f"  loading {len(existing)} cached candidates from {candidates_path}", flush=True)
        candidates = existing[:args.k_candidates]
    else:
        # (re)select up to k, pinning any existing so their cached oracles are reused
        if existing:
            print(f"  expanding from {len(existing)} -> {args.k_candidates} "
                  f"(pinning existing for cache reuse)", flush=True)
        candidates = select_candidates(wiki, seed=args.seed,
                                       n_target=args.k_candidates,
                                       force_include=existing)
        candidates = candidates[:args.k_candidates]
        save_candidates(candidates, wiki, candidates_path)

    print(f"  {len(candidates)} candidates selected", flush=True)

    # ------------------------------------------------------------------
    # Setup LLM client
    # ------------------------------------------------------------------
    llm: Optional[VLLMClient] = None
    if args.mode == "gpu":
        try:
            import urllib.request
            llm = VLLMClient(host=f"http://127.0.0.1:{args.port}", model=args.model)
            # Ping to verify
            test_resp = llm.gen("", "Say OK in one word.", max_tokens=5)
            print(f"  [llm] vLLM client connected: {test_resp[:20]}", flush=True)
        except Exception as e:
            print(f"  [warn] vLLM not available ({e}), falling back to heuristics", flush=True)
            llm = None

    # ------------------------------------------------------------------
    # Step 3: Build oracle wikis
    # ------------------------------------------------------------------
    if not args.skip_oracle:
        print("\n=== Step 3: Build oracle wikis ===", flush=True)
        oracle_labels_all: Dict[str, Dict[str, str]] = {}
        for sk in candidates:
            oracle_labels_all[sk] = build_oracle_wiki(
                sk=sk,
                wiki=wiki,
                sources=sources,
                corpus_dir=args.corpus_dir,
                llm=llm,
                oracle_dir=args.oracle_dir,
                mode=args.mode,
            )
    else:
        print("\n=== Step 3: Loading cached oracle wikis ===", flush=True)
        oracle_labels_all: Dict[str, Dict[str, str]] = {}
        for sk in candidates:
            oracle_path = os.path.join(args.oracle_dir, f"{sk}_oracle.jsonl")
            labels = {}
            if os.path.exists(oracle_path):
                with open(oracle_path) as fh:
                    for line in fh:
                        rec = json.loads(line.strip())
                        if "claim_id" in rec and "oracle_label" in rec:
                            labels[rec["claim_id"]] = rec["oracle_label"]
            if not labels:
                print(f"  [warn] no cached oracle for {sk}, building heuristic", flush=True)
                labels = build_oracle_wiki(sk, wiki, sources, args.corpus_dir, None,
                                           args.oracle_dir, args.mode)
            oracle_labels_all[sk] = labels
            n_dies = sum(1 for v in labels.values() if v == "DIES")
            print(f"  {sk}: {len(labels)} claims, {n_dies} DIES", flush=True)

    # ------------------------------------------------------------------
    # Steps 4+5: Run methods and compute metrics
    # ------------------------------------------------------------------
    print("\n=== Steps 4+5: Run partial_delete and baselines ===", flush=True)
    all_results: Dict[str, dict] = {}

    for sk in candidates:
        oracle = oracle_labels_all.get(sk, {})
        if not oracle:
            print(f"  [skip] {sk}: no oracle labels", flush=True)
            continue

        sk_results: Dict[str, dict] = {}
        print(f"\n  Processing {sk} ...", flush=True)

        # Our method: partial_delete
        print(f"    partial_delete ...", flush=True)
        t0 = time.time()
        pd_wiki = partial_delete(wiki, sk, sources, llm)
        pd_predicted = predict_labels_partial_delete(wiki, pd_wiki, mode=args.mode)
        sk_results["partial_delete"] = compute_stratum_metrics(pd_predicted, oracle, wiki, sk)
        print(f"    partial_delete done in {time.time()-t0:.1f}s "
              f"F1={sk_results['partial_delete']['all']['oracle_f1']}", flush=True)

        # Baseline 1: RAG-drop
        print(f"    rag_drop ...", flush=True)
        rag_pred = run_baseline_rag_drop(sk, wiki, sources, mode=args.mode)
        if rag_pred:
            sk_results["rag_drop"] = compute_stratum_metrics(rag_pred, oracle, wiki, sk)
        else:
            sk_results["rag_drop"] = {}

        # Baseline 2: Naive-delete
        print(f"    naive_delete ...", flush=True)
        naive_pred = run_baseline_naive_delete(sk, wiki, sources)
        sk_results["naive_delete"] = compute_stratum_metrics(naive_pred, oracle, wiki, sk)

        # Baseline 3: FLAT (skip unless GPU mode and not explicitly skipped)
        flat_pred = None
        if args.mode == "gpu" and not args.skip_flat:
            print(f"    flat ...", flush=True)
            flat_pred = run_baseline_flat(sk, wiki, sources, args.gpu_id, args.flat_dir)
        if flat_pred is not None:
            sk_results["flat"] = compute_stratum_metrics(flat_pred, oracle, wiki, sk)
        else:
            sk_results["flat"] = {"all": {"oracle_f1": None, "residue_rate": None,
                                           "over_deletion_rate": None, "change_acc": None,
                                           "n": 0, "note": "N/A"}}

        all_results[sk] = sk_results

    # ------------------------------------------------------------------
    # Step 6: Aggregate metrics + statistical tests
    # ------------------------------------------------------------------
    print("\n=== Step 6: Compute aggregate metrics ===", flush=True)

    our_f1s = []
    baseline_f1_dict: Dict[str, List[float]] = {
        "rag_drop": [],
        "naive_delete": [],
        "flat": [],
    }

    for sk, src_results in all_results.items():
        our_all = src_results.get("partial_delete", {}).get("all", {})
        if our_all.get("oracle_f1") is not None:
            our_f1s.append(float(our_all["oracle_f1"]))
        for bname in baseline_f1_dict:
            bm_all = src_results.get(bname, {}).get("all", {})
            f1 = bm_all.get("oracle_f1")
            if f1 is not None and str(f1) != "N/A":
                baseline_f1_dict[bname].append(float(f1))

    stats = compute_statistical_tests(our_f1s, baseline_f1_dict)

    print(f"\n  Our method mean Oracle-F1: {stats['our_method']['mean_f1']}", flush=True)
    print(f"  Our method 95% CI: {stats['our_method']['ci_95']}", flush=True)
    for bname in baseline_f1_dict:
        s = stats.get(bname, {})
        print(f"  {bname}: mean={s.get('mean_f1', 'N/A')} p={s.get('p', 'N/A')}", flush=True)

    # ------------------------------------------------------------------
    # Step 7: Save results
    # ------------------------------------------------------------------
    print("\n=== Step 7: Save results ===", flush=True)

    # Full results JSON
    results_path = os.path.join(args.results_dir, "e1_results.json")
    output = {
        "config": vars(args),
        "candidates": candidates,
        "per_source": all_results,
        "aggregate": {
            "our_method_f1s": our_f1s,
            "baseline_f1s": {k: v for k, v in baseline_f1_dict.items() if v},
            "stats": stats,
        },
        "elapsed_minutes": round((time.time() - t_start) / 60, 1),
    }
    with open(results_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"  saved: {results_path}", flush=True)

    # Markdown table
    table_path = os.path.join(args.results_dir, "e1_table.md")
    with open(table_path, "w") as fh:
        fh.write(format_table(all_results))
    print(f"  saved: {table_path}", flush=True)

    # Interpretation report
    report_path = os.path.join(args.results_dir, "e1_report.md")
    with open(report_path, "w") as fh:
        fh.write(format_report(all_results, stats))
    print(f"  saved: {report_path}", flush=True)

    elapsed = (time.time() - t_start) / 60
    print(f"\n=== E1 DONE in {elapsed:.1f} min ===", flush=True)
    print(f"  our method mean Oracle-F1 = {stats['our_method']['mean_f1']}", flush=True)
    if our_f1s and float(stats["our_method"]["mean_f1"]) >= 0.70:
        print("  C1 ANCHOR: PASS (Oracle-F1 >= 0.70)", flush=True)
    else:
        print("  C1 ANCHOR: REVIEW (Oracle-F1 < 0.70 or no data)", flush=True)


if __name__ == "__main__":
    main()
