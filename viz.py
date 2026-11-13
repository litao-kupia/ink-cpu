"""Turn a raw 0-255 ink TIFF into contrast-stretched PNG previews."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


def stretch(p: np.ndarray) -> np.ndarray:
    """Undo the 0.25/0.75 label-smoothing compression of the 9 um models."""
    v = (p - 0.25) / 0.5
    return np.clip(v, 0, 1)


def preview(src: Path, *, max_side: int = 1600, box: tuple[int, int, int, int] | None = None):
    a = tifffile.imread(src)
    if box is not None:
        y0, y1, x0, x1 = box
        a = a[y0:y1, x0:x1]
    p = a.astype(np.float32) / 255.0
    v = stretch(p)
    img = Image.fromarray((v * 255).astype(np.uint8))
    scale = max(img.width, img.height) / max_side
    if scale > 1:
        img = img.resize((int(img.width / scale), int(img.height / scale)), Image.LANCZOS)
    out = src.with_suffix(f".{box[0]}_{box[0] + (box[1] - box[0])}.png" if box else ".preview.png")
    img.save(out)
    print(f"{out}  {img.width}x{img.height}  "
          f"mean {v.mean():.3f} p99 {np.percentile(v, 99):.3f}")
    return out


if __name__ == "__main__":
    src = Path(sys.argv[1])
    if len(sys.argv) > 2:
        y0, y1, x0, x1 = (int(v) for v in sys.argv[2].split(","))
        preview(src, box=(y0, y1, x0, x1))
    else:
        preview(src)
