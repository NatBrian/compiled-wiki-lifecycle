"""E5 prep (CPU) — build a temporal-update substrate from HoH-QAs (external).

HoH-QAs (russwest404/HoH-QAs): each item has
  question, answer (CURRENT), evidence (CURRENT sentence),
  outdated_infos=[{answer: STALE, evidence: STALE sentence, last_modified_time}],
  document={id,title}.

We construct an evolving corpus:
  - t0 corpus: STALE evidence sentences (as short docs) + distractor evidences.
  - update stream: for each probe, the CURRENT evidence supersedes the stale one
    (same document title) -> a supersession event.
Two certified clauses on the compiled store:
  - RETENTION: current answer verifiable from the store (fidelity).
  - CURRENCY (SER): store yields the STALE answer -> staleness error. Certify SER <= eps.

Output e5_data.json {corpus0, updates, probes, distractors}. External data only.
"""
import argparse, json, os, random, re


def norm(s):
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_probes", type=int, default=160)
    ap.add_argument("--n_distractors", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "e5_data.json"))
    args = ap.parse_args()
    from datasets import load_dataset

    d = load_dataset("russwest404/HoH-QAs")["train"]
    rng = random.Random(args.seed)
    idx = list(range(min(len(d), 20000)))
    rng.shuffle(idx)

    probes, seen_titles = [], set()
    distract_pool = []
    for i in idx:
        r = d[i]
        oi = r["outdated_infos"]
        title = ""
        try:
            doc = r["document"]
            title = (doc.get("title") if isinstance(doc, dict)
                     else json.loads(doc.replace("'", '"')).get("title", "")) or ""
        except Exception:
            title = ""
        if not (isinstance(oi, list) and oi and oi[0].get("answer")
                and oi[0].get("evidence") and r.get("evidence") and r.get("answer")):
            continue
        sa, ca = norm(oi[0]["answer"]), norm(r["answer"])
        # drop exact-equal AND substring-nested pairs: nesting makes the answer-match
        # double-count (a current answer containing the stale string scores as both).
        if sa == ca or sa in ca or ca in sa or len(sa) < 3 or len(ca) < 3:
            continue
        key = title or r["question"]
        if key in seen_titles:
            distract_pool.append(r)
            continue
        seen_titles.add(key)
        if len(probes) < args.n_probes:
            probes.append({
                "question": r["question"],
                "current_answer": r["answer"], "current_evidence": r["evidence"],
                "stale_answer": oi[0]["answer"], "stale_evidence": oi[0]["evidence"],
                "title": title or f"item{i}",
            })
        else:
            distract_pool.append(r)
        if len(probes) >= args.n_probes and len(distract_pool) >= args.n_distractors:
            break

    rng.shuffle(distract_pool)
    distractors = [{"title": (p.get("document", {}) if isinstance(p.get("document"), dict)
                              else {}).get("title", f"d{j}") or f"d{j}",
                    "evidence": p["evidence"]}
                   for j, p in enumerate(distract_pool[:args.n_distractors])]

    # corpus0 = stale evidence docs + distractor docs; updates = current evidence docs
    corpus0 = {}
    updates = {}
    for k, p in enumerate(probes):
        corpus0[f"p{k}"] = {"title": p["title"], "text": p["stale_evidence"]}
        updates[f"p{k}"] = {"title": p["title"], "text": p["current_evidence"]}
    for k, dd in enumerate(distractors):
        corpus0[f"d{k}"] = {"title": dd["title"], "text": dd["evidence"]}

    json.dump({"config": vars(args), "corpus0": corpus0, "updates": updates,
               "probes": probes}, open(args.out, "w"))
    print(f"probes={len(probes)} corpus0={len(corpus0)} updates={len(updates)} "
          f"distractors={len(distractors)}", flush=True)


if __name__ == "__main__":
    main()
