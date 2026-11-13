# ink-cpu

Run the Vesuvius Challenge **9 µm cross-scroll ink models on a normal CPU**, against the
published surface volumes, with no CUDA, no `vc_reader` build, and no container.

The official flat-inference path (`vesuvius.ink_detection.inference.infer`) needs
`vc_reader`, which is a C++ extension; the ink-detection tutorial assumes Linux and an
NVIDIA GPU. This tool reproduces the same preprocessing and blending in ~200 lines of
Python so anyone with a desktop CPU can run the shared checkpoints on real scroll data.

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
```

## Which scrolls are ready to run

A surface volume is not published for every traced segment, and rendering one needs
`vc_render_tifxyz`. `scan_all_9um.py` walks the open-data catalog: of **328 traced
segments across 45 scrolls, only four scrolls publish an ≈9.36 µm render**.

| scroll | traced segments | with ≈9.36 µm render | note |
| --- | --- | --- | --- |
| PHerc0500P2 | 46 | 46 | fragment, not a sealed scroll |
| PHerc0139 | 39 | 38 | training scroll |
| PHerc0814 | 19 | 19 | training scroll |
| PHerc0841 | 3 | 3 | |

PHerc1667 and PHerc. Paris. 4 — the two most-read scrolls — publish no 9.36 µm render at
all; their 9 µm representations are pooled from 2.4 µm volumes.

The practical consequence: for a First Letters attempt on a scroll nobody has read, the
first step is still producing a surface volume (`vc_render_tifxyz`, which streams the
scroll straight from S3). This tool takes over from there. PHerc0500P2, being a fragment
with 46 ready segments and published team predictions, is the natural place to validate a
pipeline end to end.


## Caveats

- Uncompressed, full-depth chunks only. Compressed or sharded chunks are rejected with a
  clear error rather than silently misread.
- No test-time augmentation yet (`--tta-mirror` in the official tool).
- The model definition is imported from a villa checkout; this tool reimplements only
  the data path and preprocessing, not the network.
