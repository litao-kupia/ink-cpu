"""CPU ink inference for the released Vesuvius 9 um cross-scroll checkpoints.

Reads a published surface-volume OME-Zarr straight from the open-data bucket
(no vc_reader, no CUDA), tiles it, runs the checkpoint, and blends the tiles
into one probability image.

The preprocessing mirrors `vesuvius.ink_detection.inference.infer`:
  * centre-crop the depth window to the checkpoint's input depth
  * robust (percentile-clipped median/MAD) normalisation per patch
  * sigmoid on the logits, floored-Hann weighted blending
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests

BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"


# --------------------------------------------------------------------------- #
# volume access
# --------------------------------------------------------------------------- #

def _get_json(url: str) -> dict:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def open_remote_volume(base_url: str, level: int = 0) -> dict:
    """Read a Zarr v2 array's shape/chunks/dtype over plain HTTPS."""
    zarray = _get_json(f"{base_url}/{level}/.zarray")
    return {
        "shape": tuple(int(v) for v in zarray["shape"]),
        "chunks": tuple(int(v) for v in zarray["chunks"]),
        "dtype": np.dtype(zarray["dtype"]),
        "compressor": zarray.get("compressor"),
        "level": level,
    }


def _fetch_chunk(session: requests.Session, url: str) -> bytes:
    for attempt in range(4):
        try:
            r = session.get(url, timeout=120)
            if r.status_code == 404:  # sparse chunk: all fill_value
                return b""
            r.raise_for_status()
            return r.content
        except Exception:
            if attempt == 3:
                raise
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError("unreachable")


def materialise_volume(
    base_url: str,
    cache_path: Path,
    *,
    level: int = 0,
    workers: int = 32,
) -> np.ndarray:
    """Download every chunk once into a uint8 memmap and return it.

    Chunks in these volumes are full-depth (Z, 128, 128) and uncompressed, so a
    chunk maps exactly onto one tile of the inference grid.
    """
    info = open_remote_volume(base_url, level)
    depth, height, width = info["shape"]
    cz, cy, cx = info["chunks"]
    if info["compressor"] not in (None, "null"):
        raise SystemExit(
            f"compressed chunks are not supported (compressor={info['compressor']!r})"
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = cache_path.with_suffix(".json")
    meta = {"shape": [depth, height, width], "chunks": [cz, cy, cx]}
    if cache_path.exists() and meta_path.exists():
        if json.loads(meta_path.read_text()) == meta:
            print(f"volume cache hit: {cache_path}")
            return np.memmap(cache_path, dtype=info["dtype"], mode="r",
                             shape=(depth, height, width))

    volume = np.memmap(cache_path, mode="w+", dtype=info["dtype"],
                       shape=(depth, height, width))
    volume[:] = 0

    keys = [
        (z // cz, y // cy, x // cx)
        for z in range(0, depth, cz)
        for y in range(0, height, cy)
        for x in range(0, width, cx)
    ]
    print(f"downloading {len(keys)} chunks ({depth}x{height}x{width}, "
          f"{depth * height * width / 1e6:.0f} MB) with {workers} workers")

    done = 0
    empty = 0
    started = time.time()

    def work(key):
        z, y, x = key
        with requests.Session() as session:
            raw = _fetch_chunk(session, f"{base_url}/{level}/{z}/{y}/{x}")
        if not raw:
            return key, None
        return key, np.frombuffer(raw, dtype=info["dtype"]).reshape(cz, cy, cx)

    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for key, block in pool.map(work, keys):
            z, y, x = key
            if block is None:
                empty += 1
                continue
            z, y, x = z * cz, y * cy, x * cx
            d = min(cz, depth - z)
            h = min(cy, height - y)
            w = min(cx, width - x)
            volume[z:z + d, y:y + h, x:x + w] = block[:d, :h, :w]
            done += 1
            if done % 100 == 0 or done == len(keys) - empty:
                elapsed = time.time() - started
                print(f"  {done}/{len(keys)} chunks  {elapsed:6.1f}s")

    volume.flush()
    del volume
    meta_path.write_text(json.dumps(meta))
    check = np.memmap(cache_path, dtype=info["dtype"], mode="r",
                      shape=(depth, height, width))
    dense = float((np.asarray(check[depth // 2]) > 0).mean())
    print(f"cached volume at {cache_path}  (written={done}, empty={empty}, "
          f"mid-layer nonzero={dense:.3f})")
    return check


# --------------------------------------------------------------------------- #
# preprocessing / blending
# --------------------------------------------------------------------------- #

def robust_normalize(patch_zyx: np.ndarray) -> np.ndarray:
    """vc_reader's `Normalize("robust", lower=1, upper=99)` in numpy.

    Percentile-clip, centre on the median, scale by 1.4826*MAD with the
    documented fallbacks (pre-clip population std, then half the span, then 1).
    """
    x = np.asarray(patch_zyx, dtype=np.float32)
    flat = x.ravel().astype(np.float64)
    low = float(np.percentile(flat, 1.0))
    high = float(np.percentile(flat, 99.0))

    pre_std = float(flat.std())
    clipped = np.clip(x, low, high)
    center = float(np.median(clipped))
    mad = float(np.median(np.abs(clipped.astype(np.float64) - center)))

    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-6:
        scale = pre_std
    if not np.isfinite(scale) or scale < 1e-6:
        scale = (high - low) / 2.0
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return ((clipped - center) / scale).astype(np.float32, copy=False)


def hann_weight(patch: int) -> np.ndarray:
    """Floored Hann blend map, identical to infer.compute_importance_map_2d."""
    w = np.hanning(patch)  # == torch.hann_window(patch, periodic=False)
    weight = np.outer(w, w)
    weight /= max(float(weight.max()), np.finfo(np.float32).eps)
    return np.clip(weight, 0.001, None).astype(np.float32)


def sliding_positions(length: int, patch: int, stride: int) -> list[int]:
    if length <= patch:
        return [0]
    pos = list(range(0, length - patch + 1, max(1, stride)))
    if pos[-1] != length - patch:
        pos.append(length - patch)
    return pos


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #

def pool_depth(
    volume: np.ndarray,
    *,
    pool: int = 4,
    out_z: int = 21,
    block: int = 512,
) -> np.ndarray:
    """Reproduce `prepare_9um_isotropic_input.prepare_isotropic_input`.

    Centre-crop the Z axis to `out_z * pool` planes, then average each group of
    `pool` planes with round-half-to-even, giving `out_z` slices. This is how the
    released 9 um models turn a 2.4 um surface volume (read at pyramid level 2)
    into the representation they were trained on.
    """
    import math

    depth, height, width = volume.shape
    need = out_z * pool
    if depth < need:
        raise SystemExit(
            f"cannot pool {depth} planes into {out_z}x{pool} = {need} planes"
        )
    z0 = math.ceil((depth - need) / 2)
    src = volume[z0:z0 + need]
    print(f"pooling Z: {depth} planes -> {out_z} slices "
          f"(source planes {z0}..{z0 + need - 1}, mean of {pool})")

    out = np.empty((out_z, height, width), dtype=np.uint8)
    for y0 in range(0, height, block):
        y1 = min(height, y0 + block)
        for x0 in range(0, width, block):
            x1 = min(width, x0 + block)
            block_zyx = np.asarray(src[:, y0:y1, x0:x1], dtype=np.float32)
            block_zyx = block_zyx.reshape(out_z, pool, y1 - y0, x1 - x0)
            out[:, y0:y1, x0:x1] = np.rint(block_zyx.mean(axis=1)).astype(np.uint8)
    return out


def build_model(checkpoint_path: Path, villa_src: str | None):
    """Rebuild the checkpoint's model from the villa source tree.

    The model definition lives in `vesuvius`; only pure-Python modules are used
    (no vc_reader), so pointing at a checkout is enough.
    """
    src = villa_src or os.environ.get("VILLA_SRC")
    if not src:
        src = str(Path(__file__).resolve().parent / "villa" / "vesuvius" / "src")
    if not Path(src).is_dir():
        raise SystemExit(
            "set --villa-src or $VILLA_SRC to <villa>/vesuvius/src "
            f"(tried {src})"
        )
    if src not in sys.path:
        sys.path.insert(0, src)

    from vesuvius.ink_detection.models.checkpoint import (
        config_from_checkpoint,
        load_checkpoint,
        select_inference_weights,
    )
    from vesuvius.ink_detection.models.model import make_model

    payload = load_checkpoint(checkpoint_path)
    _, state = select_inference_weights(payload, source=checkpoint_path)
    config = config_from_checkpoint(payload, source=checkpoint_path)
    model = make_model(config)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise SystemExit(
            f"checkpoint mismatch: {len(incompatible.missing_keys)} missing, "
            f"{len(incompatible.unexpected_keys)} unexpected"
        )
    model.eval()
    crop_z, crop_y, crop_x = config.model.crop_size
    if crop_y != crop_x:
        raise SystemExit(f"non-square patch in checkpoint: {config.model.crop_size}")
    return model, int(crop_y), int(crop_z)


def run(
    volume: np.ndarray,
    models,
    *,
    patch: int,
    input_depth: int,
    overlap: float,
    batch_size: int,
    reverse_depth: bool,
    roi: tuple[int, int, int, int] | None,
    threads: int,
    layer_offset: int = 0,
    tta_mirror: bool = False,
    progress_every: int = 50,
) -> tuple[np.ndarray, dict]:
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(threads)
    torch.set_grad_enabled(False)

    depth, height, width = volume.shape
    y0, y1, x0, x1 = roi or (0, height, 0, width)
    roi_h, roi_w = y1 - y0, x1 - x0

    # centre-crop the depth window the checkpoint expects (infer.select_layer_indices),
    # then shift it: the models jittered this window during training and are sensitive
    # to where the surface sits inside it.
    if input_depth < depth:
        start = depth // 2 - input_depth // 2 + layer_offset
        start = max(0, min(start, depth - input_depth))
        layers = np.arange(start, start + input_depth)
    else:
        layers = np.arange(depth)
    if reverse_depth:
        layers = layers[::-1]

    stride = max(1, int(round(patch * (1.0 - overlap))))
    ys = sliding_positions(roi_h, patch, stride)
    xs = sliding_positions(roi_w, patch, stride)

    prob_sum = np.zeros((roi_h, roi_w), dtype=np.float32)
    weight_sum = np.zeros((roi_h, roi_w), dtype=np.float32)
    weight = hann_weight(patch)

    coords = [(y, x) for y in ys for x in xs]
    total = len(coords)
    print(f"tiles: {total} ({len(ys)}x{len(xs)}), patch={patch}, "
          f"stride={stride}, depth window={layers[0]}..{layers[-1]}")

    started = time.time()
    processed = 0
    for start in range(0, total, batch_size):
        batch_coords = coords[start:start + batch_size]
        tensors = []
        metas = []
        for (yy, xx) in batch_coords:
            gy0, gx0 = y0 + yy, x0 + xx
            valid_h = min(patch, height - gy0)
            valid_w = min(patch, width - gx0)
            patch_zyx = np.zeros((input_depth, patch, patch), dtype=np.uint8)
            src = volume[layers, gy0:gy0 + valid_h, gx0:gx0 + valid_w]
            patch_zyx[:, :valid_h, :valid_w] = src
            norm = robust_normalize(patch_zyx)
            tensors.append(torch.from_numpy(norm).unsqueeze(0))
            metas.append((yy, xx, valid_h, valid_w))

        batch = torch.stack(tensors)
        probs_acc = None
        for model in models:
            variants = [batch]
            if tta_mirror:
                # average the identity with the two axis mirrors and their product
                variants = [batch, torch.flip(batch, [-1]), torch.flip(batch, [-2]),
                            torch.flip(batch, [-2, -1])]
            model_probs = None
            for v, tensor in enumerate(variants):
                logits = model(tensor)
                if isinstance(logits, dict):
                    logits = logits["ink"]
                p = logits.float().sigmoid()
                if p.shape[-2:] != (patch, patch):
                    p = F.interpolate(p, size=(patch, patch), mode="bilinear",
                                      align_corners=False)
                if v == 1:
                    p = torch.flip(p, [-1])
                elif v == 2:
                    p = torch.flip(p, [-2])
                elif v == 3:
                    p = torch.flip(p, [-2, -1])
                model_probs = p if model_probs is None else model_probs + p
            model_probs = model_probs / float(len(variants))
            probs_acc = model_probs if probs_acc is None else probs_acc + model_probs
        probs = (probs_acc / float(len(models))).squeeze(1).numpy()

        for p, (yy, xx, vh, vw) in zip(probs, metas):
            prob_sum[yy:yy + vh, xx:xx + vw] += p[:vh, :vw] * weight[:vh, :vw]
            weight_sum[yy:yy + vh, xx:xx + vw] += weight[:vh, :vw]

        processed += len(batch_coords)
        if processed % progress_every < batch_size or processed == total:
            elapsed = time.time() - started
            rate = processed / max(elapsed, 1e-9)
            eta = (total - processed) / max(rate, 1e-9)
            print(f"  {processed}/{total} tiles  {rate:5.2f} tile/s  "
                  f"elapsed {elapsed:6.1f}s  eta {eta:6.1f}s")

    out = prob_sum / np.maximum(weight_sum, 1e-6)
    stats = {
        "tiles": total,
        "seconds": round(time.time() - started, 1),
        "patch": patch,
        "stride": stride,
        "depth_layers": [int(layers[0]), int(layers[-1])],
        "layer_offset": int(layer_offset),
        "models": len(models),
        "tta_mirror": bool(tta_mirror),
        "p50": round(float(np.median(out)), 4),
        "p99": round(float(np.percentile(out, 99)), 4),
        "p999": round(float(np.percentile(out, 99.9)), 4),
        "frac_gt_0.6": round(float((out > 0.6).mean()), 5),
    }
    return out, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zarr", required=True,
                    help="surface-volume zarr: https://..., s3://..., or local path")
    ap.add_argument("--checkpoint", required=True, nargs="+",
                    help="one or more checkpoints; averaged as an ensemble")
    ap.add_argument("--out", required=True, help="output TIFF (uint8 0-255)")
    ap.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "ink-cpu"))
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=32, help="chunk download workers")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--roi", default=None, help="y0,y1,x0,x1 to restrict inference")
    ap.add_argument("--direction", choices=("forward", "reverse"), default="forward")
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--villa-src", default=None,
                    help="path to <villa>/vesuvius/src (or set VILLA_SRC)")
    ap.add_argument("--z-pool", type=int, default=1,
                    help="mean-pool this many depth planes (4 for 2.4 um level-2 renders)")
    ap.add_argument("--pooled-z", type=int, default=21,
                    help="slices produced by --z-pool (the 9 um recipe uses 21)")
    ap.add_argument("--layer-offset", type=int, default=0,
                    help="shift the depth window by this many slices")
    ap.add_argument("--tta-mirror", action="store_true",
                    help="average the four axis mirrors of each patch")
    args = ap.parse_args(argv)

    zarr = args.zarr
    if zarr.startswith("s3://"):
        zarr = BUCKET + "/" + zarr[len("s3://"):].split("/", 1)[1]
    if zarr.startswith("http"):
        base = zarr.rstrip("/")
        name = base.split("/")[-1] or "volume"
        cache = Path(args.cache_dir) / name / f"volume.level{args.level}.u8"
        volume = materialise_volume(base, cache, level=args.level,
                                    workers=args.workers)
    else:
        import zarr
        volume = zarr.open(args.zarr, mode="r")[str(args.level)]

    if args.z_pool > 1:
        volume = pool_depth(volume, pool=args.z_pool, out_z=args.pooled_z)

    models = []
    patch = input_depth = None
    for ckpt in args.checkpoint:
        model, ck_patch, ck_depth = build_model(Path(ckpt), args.villa_src)
        if patch is not None and (ck_patch, ck_depth) != (patch, input_depth):
            raise SystemExit("checkpoints disagree on patch size or input depth")
        patch, input_depth = ck_patch, ck_depth
        models.append(model)
    print(f"ensemble: {len(models)} checkpoint(s), tta_mirror={args.tta_mirror}")

    roi = tuple(int(v) for v in args.roi.split(",")) if args.roi else None

    result, stats = run(
        volume, models,
        patch=patch, input_depth=input_depth,
        overlap=args.overlap, batch_size=args.batch_size,
        reverse_depth=args.direction == "reverse",
        roi=roi, threads=args.threads,
        layer_offset=args.layer_offset,
        tta_mirror=args.tta_mirror,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import tifffile
    tifffile.imwrite(out_path, np.clip(result * 255.0, 0, 255).astype(np.uint8))
    out_path.with_suffix(".json").write_text(json.dumps(
        {**stats, "zarr": args.zarr, "checkpoints": args.checkpoint}, indent=1))
    print(f"wrote {out_path}  {result.shape}  {json.dumps(stats)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
