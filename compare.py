"""Compare a CPU 9 um prediction against a published team prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


def load(path: Path) -> np.ndarray:
    a = tifffile.imread(path)
    if a.ndim > 2:
        a = a[..., 0]
    return a.astype(np.float32) / 255.0


def main() -> int:
    ours_path = Path(sys.argv[1])
    ref_path = Path(sys.argv[2])

    ours = (load(ours_path) - 0.25) / 0.5
    ours = np.clip(ours, 0, 1)
    ref = load(ref_path)
    print(f"ours {ours.shape}  ref {ref.shape}")

    if ours.shape != ref.shape:
        ref_img = Image.fromarray((np.clip(ref, 0, 1) * 255).astype(np.uint8))
        ref_img = ref_img.resize((ours.shape[1], ours.shape[0]), Image.BILINEAR)
        ref = np.asarray(ref_img, dtype=np.float32) / 255.0

    mask = ref > 0.02
    r = float(np.corrcoef(ours[mask].ravel(), ref[mask].ravel())[0, 1])
    print(f"masked pixels {int(mask.sum()):,}  pearson r = {r:.4f}")
    print(f"ours mean {ours.mean():.3f}  ref mean {ref.mean():.3f}")

    h, w = ours.shape
    box = (h // 2 - 600, h // 2 + 600, w // 2 - 600, w // 2 + 600)
    y0, y1, x0, x1 = box
    side = np.concatenate([ours[y0:y1, x0:x1], ref[y0:y1, x0:x1]], axis=1)
    out = ours_path.with_suffix(".compare.png")
    Image.fromarray((np.clip(side, 0, 1) * 255).astype(np.uint8)).save(out)
    print(f"wrote {out}  (left: this tool, right: published)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
