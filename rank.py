import glob
import json
import os

rows = []
for p in glob.glob("out/sweep/PHerc0009B/*.json"):
    if os.path.basename(p).startswith("summary"):
        continue
    rows.append(json.load(open(p)))

rows.sort(key=lambda r: -r["frac_gt_0.6"])
print(f"{len(rows)} segments screened")
for r in rows:
    print(f"  {r['name'][:44]:46} frac>0.6={r['frac_gt_0.6']:.4f}  p99={r['p99']:.3f}")
