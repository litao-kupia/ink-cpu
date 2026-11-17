import glob
import json

rows = [json.load(open(p)) for p in glob.glob("out/sweep/PHerc0009B/*.json")
        if "summary" not in p]
rows.sort(key=lambda r: -r["frac_gt_0.6"])
tiles = sum(r["tiles"] for r in rows)
secs = sum(r["seconds"] for r in rows)
print(f"segments={len(rows)} tiles={tiles:,} inference={secs / 60:.0f} min")
print(f"frac>0.6 range {rows[-1]['frac_gt_0.6']:.4f}..{rows[0]['frac_gt_0.6']:.4f}")
print(f"p99 range {min(r['p99'] for r in rows):.3f}..{max(r['p99'] for r in rows):.3f}")
for r in rows[:5]:
    print(f"  {r['name'][:46]:48} {r['frac_gt_0.6']:.4f}")
