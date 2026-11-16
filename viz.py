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


def grid(src: Path, cols: int = 4, rows: int = 3) -> None:
    """Write native-resolution tiles so a big prediction can actually be read."""
    a = np.clip((tifffile.imread(src).astype(np.float32) / 255.0 - 0.25) / 0.5, 0, 1)
    h, w = a.shape
    outdir = src.parent / (src.stem + "_tiles")
    outdir.mkdir(exist_ok=True)
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * h // rows, (r + 1) * h // rows
            x0, x1 = c * w // cols, (c + 1) * w // cols
            path = outdir / f"r{r}c{c}.png"
            Image.fromarray((a[y0:y1, x0:x1] * 255).astype(np.uint8)).save(path)
    print(f"wrote {cols * rows} tiles to {outdir}")


if __name__ == "__main__":
    src = Path(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2] == "--grid":
        grid(src, int(sys.argv[3]) if len(sys.argv) > 3 else 4,
             int(sys.argv[4]) if len(sys.argv) > 4 else 3)
    elif len(sys.argv) > 2:
        y0, y1, x0, x1 = (int(v) for v in sys.argv[2].split(","))
        preview(src, box=(y0, y1, x0, x1))
    else:
        preview(src)
