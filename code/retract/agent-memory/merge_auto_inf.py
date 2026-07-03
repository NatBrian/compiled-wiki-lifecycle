import json, glob, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parts = sorted(glob.glob(f"{ROOT}/results/e0b_auto_inf_part_*.json"))
recs = []
for p in parts:
    recs += json.load(open(p))["per_fact"]
seen = {}
for r in recs:
    seen[r["id"]] = r
recs = [seen[k] for k in sorted(seen)]
arms = ["A1_none", "A3_hash", "A3_cone", "A3_nli", "A4"]
def wilson(k, n, z=1.96):
    if not n: return [None, None]
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); h=z*((p*(1-p)+z*z/(4*n))/n)**0.5
    return [round(max(0.0,(c-h)/d),4), round((c+h)/d,4)]
summ={"n_facts":len(recs),"n_consolid":2,"cone_tau":0.55,"arms":{}}
for a in arms:
    xs=[r[a] for r in recs if a in r and "backflowed" in r[a]]
    n=len(xs); bf=sum(x["backflowed"] for x in xs)
    summ["arms"][a]={"n":n,"leak":round(bf/n,4) if n else None,"leak_wilson95":wilson(bf,n)}
h=summ["arms"]["A3_hash"]["leak"]; nli=summ["arms"]["A3_nli"]["leak"]; none=summ["arms"]["A1_none"]["leak"]
summ["native_inference_backflow"]=bool(none and none>=0.10)
summ["hash_insufficient"]=bool(h is not None and nli is not None and (h-nli)>=0.10)
summ["semantic_necessary_native"]=bool(summ["native_inference_backflow"] and summ["hash_insufficient"])
json.dump({"summary":summ,"per_fact":recs}, open(f"{ROOT}/results/e0b_auto_inf.json","w"), indent=2)
print(json.dumps(summ,indent=1))
