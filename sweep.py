"""Screen many segments of one scroll: one model, forward, score, rank.

Usage:
  python sweep.py --scroll PHerc0009B --res 2.401 --out-dir out/sweep
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tifffile

import ink_cpu

S3 = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scroll", required=True)
    ap.add_argument("--res", default="2.401", help="resolution tag to match, e.g. 2.401")
    ap.add_argument("--checkpoint", nargs="+", required=True)
    ap.add_argument("--villa-src", default=None)
    ap.add_argument("--out-dir", default="out/sweep")
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--z-pool", type=int, default=4)
    ap.add_argument("--direction", choices=("forward", "reverse"), default="forward")
    ap.add_argument("--layer-offset", type=int, default=0)
    ap.add_argument("--tta-mirror", action="store_true")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    table = json.load(open("surface_volumes.json"))
    items = []
    for seg, vols in table.items():
        if not seg.startswith(args.scroll + "/"):
            continue
        for v in vols:
            if args.res in v:
                items.append((seg, v))
                break
    items.sort()
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} segments of {args.scroll} with a {args.res} um render")

    out_dir = Path(args.out_dir) / args.scroll
    out_dir.mkdir(parents=True, exist_ok=True)

    models = []
    patch = depth = None
    for ckpt in args.checkpoint:
        m, p, d = ink_cpu.build_model(Path(ckpt), args.villa_src)
        if patch is not None and (p, d) != (patch, depth):
            raise SystemExit("checkpoints disagree on patch/depth")
        patch, depth = p, d
        models.append(m)

    rows = []
    for i, (seg, vol) in enumerate(items, 1):
        name = seg.split("/segments/")[1].strip("/").replace("/", "_")
        tif = out_dir / f"{name}.tif"
        meta = out_dir / f"{name}.json"
        if tif.exists() and meta.exists():
            rows.append(json.loads(meta.read_text()))
            print(f"[{i}/{len(items)}] {name}: cached")
            continue

        url = f"{S3}/{vol}".rstrip("/")
        t0 = time.time()
        # reuse ink_cpu's cache convention so earlier downloads are not repeated
        vol_name = url.split("/")[-1] or "volume"
        cache = (Path.home() / ".cache" / "ink-cpu" / vol_name /
                 f"volume.level{args.level}.u8")
        volume = ink_cpu.materialise_volume(url, cache, level=args.level, workers=32)
        if args.z_pool > 1:
            volume = ink_cpu.pool_depth(volume, pool=args.z_pool)

        result, stats = ink_cpu.run(
            volume, models, patch=patch, input_depth=depth,
            overlap=0.5, batch_size=args.batch_size,
            reverse_depth=args.direction == "reverse",
            roi=None, threads=args.threads,
            layer_offset=args.layer_offset, tta_mirror=args.tta_mirror,
            progress_every=100000,
        )
        tifffile.imwrite(tif, np.clip(result * 255.0, 0, 255).astype(np.uint8))
        row = {"segment": seg, "volume": url, "name": name, **stats,
               "wall_seconds": round(time.time() - t0, 1),
               "direction": args.direction, "layer_offset": args.layer_offset}
        meta.write_text(json.dumps(row, indent=1))
        rows.append(row)
        print(f"[{i}/{len(items)}] {name}: p99={stats['p99']:.3f} "
              f"frac>0.6={stats['frac_gt_0.6']:.4f} {stats['seconds']:.0f}s")
        del volume

    rows.sort(key=lambda r: -r["frac_gt_0.6"])
    summary = out_dir / f"summary_{args.direction}_off{args.layer_offset}.json"
    summary.write_text(json.dumps(rows, indent=1))
    print(f"\ntop by frac>0.6 ({summary}):")
    for r in rows[:8]:
        print(f"  {r['name'][:44]:46} {r['frac_gt_0.6']:.4f}  p99={r['p99']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
