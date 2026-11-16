"""Compare a pooled 2.4 um ROI against the native 9.36 um full render."""

import numpy as np
import tifffile
from PIL import Image

pooled = tifffile.imread("out/w035_24um_pooled.tif").astype(np.float32) / 255.0
native = tifffile.imread("out/w035_full.tif").astype(np.float32) / 255.0

# the two renders differ slightly in size (5660x5100 vs 5820x5240)
sy = native.shape[0] / 5660
sx = native.shape[1] / 5100
y0, x0, side = 1800, 1800, 1200
ny0, nx0 = int(y0 * sy), int(x0 * sx)
ny1, nx1 = int((y0 + side) * sy), int((x0 + side) * sx)
crop = native[ny0:ny1, nx0:nx1]
crop = np.asarray(Image.fromarray((crop * 255).astype(np.uint8)).resize(
    (side, side), Image.BILINEAR), dtype=np.float32) / 255.0

a = np.clip((pooled - 0.25) / 0.5, 0, 1)
b = np.clip((crop - 0.25) / 0.5, 0, 1)
print(f"pooled {a.shape}  native crop {b.shape}")
print(f"pearson r = {np.corrcoef(a.ravel(), b.ravel())[0, 1]:.3f}")
print(f"pooled mean {a.mean():.3f}  native mean {b.mean():.3f}")

side_by_side = np.concatenate([a, b], axis=1)
Image.fromarray((side_by_side * 255).astype(np.uint8)).save("out/pooled_vs_native.png")
print("wrote out/pooled_vs_native.png (left: 2.4 um pooled, right: native 9.36 um)")
