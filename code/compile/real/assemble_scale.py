"""Assemble the SCALING-phase results into markdown tables with Wilson 95% CIs:

  1. Pooled-HoH currency-at-scale (n=300 shared 628-doc corpus): SER + acc per arm,
     including the real published LightRAG. The headline crossover.
  2. LOFT accuracy-at-scale control (no staleness): dump vs RAG at 32k/128k.
  3. Reader-size sweep: SER per arm at 1.5B / 7B / 14B readers.

Reads whatever result files exist; missing ones are skipped (so it can be run
mid-experiment). Writes results/results_scale_summary.json.
"""
import json, math, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"


def wilson(k, n, z=1.96):
    if not n: return (None, None)
    p = k/n; d = 1+z*z/n
    c = (p+z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (round(max(0,c-h),3), round(min(1,c+h),3))


def load(tag):
    p = R/f"results_{tag}.json"
    return json.loads(p.read_text()) if p.exists() else None


MECH = {"closed_book":"parametric prior", "full_dump_cic":"long-context dump (no curation)",
        "vector_rag":"RAG accumulate", "lightrag":"graph-RAG accumulate",
        "resolve_free":"label-free recency resolve", "wiki_karpathy":"wiki overwrite"}
ORDER = ["closed_book","full_dump_cic","vector_rag","lightrag","resolve_free","wiki_karpathy"]


def pooled_table():
    print("\n=== 1. POOLED-HoH currency-at-scale (n=300, shared 628-doc corpus) ===")
    print(f"{'system':<22}{'mechanism':<34}{'SER':>7}{'95% CI':>16}{'acc':>7}")
    print("-"*86)
    out = []
    for a in ORDER:
        d = load(f"pool_{a}")
        if not d: continue
        s = d["summary"]; n=s["n"]; st=s["n_stale"]
        lo,hi = wilson(st,n)
        ci = f"[{lo:.2f},{hi:.2f}]" if lo is not None else "n/a"
        print(f"{a:<22}{MECH.get(a,''):<34}{s['ser']:>7.3f}{ci:>16}{s['accuracy']:>7.3f}")
        out.append({"system":a,"mechanism":MECH.get(a,""),"ser":s["ser"],"ser_ci":[lo,hi],
                    "acc":s["accuracy"],"n":n,"n_stale":st})
    return out


def loft_table():
    rows=[]
    for ds in ("nq","hotpotqa"):
        for sc in ("32k","128k"):
            for arm in ("full_dump_cic","vector_rag"):
                d = load(f"loft_{arm}_{ds}_{sc}")
                if d: rows.append(d)
    if not rows: return []
    print("\n=== 2. LOFT accuracy-at-scale (no staleness; dump vs RAG) ===")
    print(f"{'dataset':<10}{'scale':<7}{'arm':<16}{'acc':>7}{'corpus':>8}{'n':>4}")
    print("-"*52)
    for d in rows:
        print(f"{d['dataset']:<10}{d['scale']:<7}{d['arm']:<16}{d['accuracy']:>7.3f}{d['n_corpus']:>8}{d['n']:>4}")
    return [{"dataset":d["dataset"],"scale":d["scale"],"arm":d["arm"],
             "acc":d["accuracy"],"n_corpus":d["n_corpus"],"n":d["n"]} for d in rows]


def sweep_table():
    sizes = [("_1p5b","1.5B"),("_7b","7B"),("","14B")]
    arms = ["full_dump_cic","vector_rag","resolve_free","wiki_karpathy"]
    present = {(a,sfx):load(f"pool_{a}{sfx}") for a in arms for sfx,_ in sizes}
    if not any(present.values()): return []
    print("\n=== 3. Reader-size sweep on pooled-HoH (SER; lower=better) ===")
    print(f"{'arm':<16}" + "".join(f"{lab:>10}" for _,lab in sizes))
    print("-"*(16+10*len(sizes)))
    out=[]
    for a in arms:
        cells=[]
        for sfx,_ in sizes:
            d = present[(a,sfx)]
            cells.append(f"{d['summary']['ser']:.3f}" if d else "  -")
        print(f"{a:<16}" + "".join(f"{c:>10}" for c in cells))
        out.append({"arm":a,"ser_by_size":{lab:(present[(a,sfx)]['summary']['ser'] if present[(a,sfx)] else None) for sfx,lab in sizes}})
    return out


def main():
    summary = {"pooled_hoh": pooled_table(), "loft": loft_table(), "size_sweep": sweep_table()}
    (R/"results_scale_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {R/'results_scale_summary.json'}")


if __name__ == "__main__":
    main()
