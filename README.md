# ink-cpu

Run the Vesuvius Challenge **9 µm cross-scroll ink models on a normal CPU**, against the
published surface volumes, with no CUDA, no `vc_reader` build, and no container.

The official flat-inference path (`vesuvius.ink_detection.inference.infer`) needs
`vc_reader`, which is a C++ extension; the ink-detection tutorial assumes Linux and an
NVIDIA GPU. This tool reproduces the same preprocessing and blending in ~200 lines of
Python so anyone with a desktop CPU can run the shared checkpoints on real scroll data.

Originally built for the November 2026 progress prize round; `SUBMISSION.md` records the
measurements and results on real scrolls.

## What it does

1. Streams a published surface-volume OME-Zarr straight from the open-data bucket
   (`https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com`) into a local
   `uint8` cache. Chunks in these volumes are full-depth 128×128 and uncompressed, so
   one chunk is exactly one inference tile.
2. Rebuilds the checkpoint's model from a villa checkout (pure-Python modules only).
3. Tiles the segment and blends with a floored Hann window, exactly as `infer.py` does:

   | step | behaviour |
   | --- | --- |
   | depth window | centre-crop to the checkpoint's input depth (`17` slices of 28) |
   | normalisation | `robust_mad`: clip to p1–p99, centre on median, scale by 1.4826·MAD |
   | blending | `hann`, floored at `0.001`, 50 % overlap by default |
   | output | sigmoid probabilities × 255 as a grayscale TIFF |

## Usage

```bash
uv venv --system-site-packages .venv      # reuse an existing torch CPU install
uv pip install -r requirements.txt

python ink_cpu.py \
  --zarr  https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/segments/20260317000000-w035_2026031718/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr \
  --checkpoint checkpoints/hybrid_3d2d-seed42_step-075000.pth \
  --villa-src ../villa/vesuvius/src \
  --out out/w035.tif \
  --batch-size 4
```

`--roi y0,y1,x0,x1` restricts inference to a sub-region. `--direction reverse` runs the
depth window backwards, which matters when you do not know which side of the surface
volume faces the ink. `--overlap` defaults to `0.5`.

### 2.4 µm surface volumes

For the many segments that only publish a 2.4 µm render, read pyramid level 2 and pool
the depth axis with the released recipe (centre-crop to 21×4 planes, average each group
of four, round to uint8):

```bash
python ink_cpu.py \
  --zarr .../2.399um-<...>.zarr --level 2 --z-pool 4 \
  --checkpoint checkpoints/hybrid_3d2d-seed42_step-075000.pth \
  --villa-src ../villa/vesuvius/src \
  --out out/segment.tif
```

This reproduces `vesuvius.ink_detection.preprocessing.prepare_9um_isotropic_input`
without writing an intermediate Zarr.

Checkpoints come from [`scrollprize/ink_9um`](https://huggingface.co/scrollprize/ink_9um):

```bash
hf download scrollprize/ink_9um hybrid_3d2d-seed42/step-075000.pth --local-dir checkpoints
```

## Measured throughput

Intel Core Ultra 9 285K (24 cores, 127 GB RAM, no GPU), `torch 2.13.0+cpu`,
`--batch-size 4 --threads 24 --overlap 0.5`:

| workload | tiles | time | rate |
| --- | --- | --- | --- |
| 256×256 ROI | 9 | 0.5 s | — |
| w035 full segment (5820×5240) | 7 290 | 388 s | 18.8 tile/s |
| PHerc0500P2 segment (4520×3120) | 3 360 | 186 s | 18.0 tile/s |
| w035 ROI via 2.4 µm level-2 + pooling | 324 | 18.3 s | 17.7 tile/s |

Downloading is the slow part: 854 MB for a native 9.4 µm render, 3.1 GB for a 2.4 µm
render at level 2 (both cached, both about two to five minutes with 32 workers).

Downloading the w035 volume (854 MB, 1 623/1 886 chunks) from S3 takes about two minutes
with 32 workers and is cached for re-runs.

For comparison, the same segment takes on the order of a few minutes on an H100 with the
official container. A desktop CPU is therefore roughly an order of magnitude slower, not
orders — which is the point: a GPU is not a prerequisite for trying a scroll.


## Reading the output

The 9 µm models are trained with label smoothing 0.5, so background sits near 0.25 and
ink near 0.75. Raw predictions look washed out; rescale with `(p - 0.25) / 0.5` before
judging them. `viz.py` does that and writes PNG previews:

```bash
python viz.py out/w035.tif
python viz.py out/w035.tif --grid 3 3      # native-resolution tiles, easier to read
```

## Hunting harder: ensembles, TTA, depth offsets

Weak predictions are usually a model or depth problem, not a threshold problem. The tool
can average checkpoints, average the four axis mirrors, and shift the depth window:

```bash
python ink_cpu.py \
  --checkpoint checkpoints/hybrid_3d2d-seed42_step-075000.pth \
               checkpoints/hybrid_3d2d-seed43_step-075000.pth \
  --tta-mirror --layer-offset 1 \
  --zarr ... --level 2 --z-pool 4 --out out/segment.tif
```

`--layer-offset` shifts the 17-slice window inside the volume. The released models were
trained with a ±2-plane jitter of that window, so ±2 is the meaningful range.

`sweep.py` screens every segment of a scroll with one model, records `p99` and
`frac>0.6` per segment into a JSON summary, and ranks them — so the expensive
ensemble/TTA runs go only to the promising segments:

```bash
python sweep.py --scroll PHerc0009B --res 2.401 \
  --checkpoint checkpoints/hybrid_3d2d-seed42_step-075000.pth --out-dir out/sweep
```

## Which scrolls are ready to run

`scan_surface_volumes.py` walks every traced segment in the open-data catalog and records
which surface volumes are published, by resolution. Of **328 traced segments across 45
scrolls**, two paths are usable today:

| representation | segments | scrolls |
| --- | --- | --- |
| native ≈9.4 µm render | 106 | PHerc0139 (38), PHerc0814 (19), PHerc0841 (3), PHerc0500P2 (46) |
| 2.4 µm render, `--level 2 --z-pool 4` | 196 | PHerc. Paris. 4 (80), PHerc0139 (38), PHerc1667 (38), PHerc0814 (19), PHerc0009B (18), PHerc0841 (3) |

The 2.4 µm path is what unlocks the two most-read scrolls: neither PHerc1667 nor
PHerc. Paris. 4 publishes a 9 µm render, but both publish 2.4 µm renders that pool
exactly into the representation the models were trained on.

Everything else — 7.9 µm DLS-era scans, 8.64 µm renders, unrendered segments — is not
directly usable without a re-render.



## Caveats

- Uncompressed, full-depth chunks only. Compressed or sharded chunks are rejected with a
  clear error rather than silently misread.
- No test-time augmentation yet (`--tta-mirror` in the official tool).
- The model definition is imported from a villa checkout; this tool reimplements only
  the data path and preprocessing, not the network.
