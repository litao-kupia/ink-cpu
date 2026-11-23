# Progress Prize submission — ink-cpu

**One line:** the shared 9 µm cross-scroll ink models now run on ordinary CPU machines,
against the published surface volumes, with no CUDA and no `vc_reader` build.

## What was built

`ink-cpu` is a small, dependency-light runner for the released
[`scrollprize/ink_9um`](https://huggingface.co/scrollprize/ink_9um) checkpoints. It
reproduces the official flat-inference contract in pure Python:

- streams a published surface-volume OME-Zarr from the open-data bucket into a local
  `uint8` cache, using the fact that these chunks are full-depth 128×128 and uncompressed,
  so one chunk is exactly one inference tile;
- centre-crops the depth window the checkpoint expects (17 of 28 slices);
- applies the same `robust_mad` normalisation (percentile clip, median centre, 1.4826·MAD
  scale, including the documented fallbacks);
- tiles with 50 % overlap and blends with the same floored Hann window;
- writes a grayscale probability TIFF, and `viz.py` rescales it the way the models expect
  (background ≈ 0.25, ink ≈ 0.75).

Roughly 200 lines. `torch` CPU, `numpy`, `tifffile`, `requests`.

## Why it matters

The ink-detection tutorial states it plainly: it is written for Linux and assumes an
NVIDIA GPU. The official flat-inference script imports `vc_reader`, a C++ extension that
has to be built from source. That shuts out anyone on a laptop, and WSL2 does not help
without a GPU anyway.

The result is that the cheapest, most accessible way to *look for ink in a scroll nobody
has read* — render a segment at 9 µm and run the shared checkpoints — has had a hardware
gate in front of it. This removes the gate.

## Evidence on real data

### Validation segment: PHerc0139 `w035` (in the models' training set)

Full segment, 5820×5240, 7 290 tiles at 18.8 tiles/s on a 24-core desktop, no GPU:
**388 s end to end**. Output shows clean, correctly oriented Greek letterforms with no
tiling seams. This is the expected control: the pipeline is faithful enough to reproduce
legible text on a segment the models know.

### Used for real: screening two unread scrolls

The tool was pointed at scrolls where nothing has been read, to answer a question that only
matters if it is asked on unseen data: does the released cross-scroll model actually fire
there?

| scroll | segments | tiles | `frac>0.6` range |
| --- | --- | --- | --- |
| PHerc0009B | 18 (2.4 µm, level-2 pooled) | 121 383 | 0.0154 – 0.0451 |
| PHerc0841 | 3 (native 9.37 µm) | 14 474 | 0.0307 – 0.0436 |

115 minutes of inference for PHerc0009B, about 13 for PHerc0841, 56.6 GB streamed from S3
in total. Both scrolls behave identically: flat, low scores with no segment standing out,
and native-resolution inspection turns the apparent letterforms into blobs and papyrus
weave rather than strokes. The best PHerc0009B segment was re-run with the two-seed
ensemble and a +1 depth offset; it stayed flat, and `frac>0.6` *fell* to 0.0298 — averaging
made it less confident, not more, which is what happens when there is no signal to
reinforce.

Result: **no legible text in 21 segments across two unread scrolls.**

That is a negative result and it is reported as one. It also comes with a caveat that
matters: 15 of the 18 PHerc0009B segments are `auto_grown` debug meshes, so flat output
there is partly a statement about segment quality, not only about the model. The three
segments that are not `auto_grown` were flat too, but three is thin evidence. What can be
said safely is narrower than "the model does not generalise": on these particular traced
surfaces, it produced structure and no strokes, and the ensemble reduced rather than
raised its confidence.

The cost result is not affected by that caveat. Screening a whole scroll is now two hours
on a desktop CPU instead of a GPU job, so "is there anything here?" can be answered before
committing to a labelling effort.



### Second check: PHerc0500P2 fragment

This fragment is the only non-training sample with a full set of published ≈9.36 µm
renders (46 of its 46 traced segments), so it is the natural place to test a pipeline
against data the models did not train on.

A full segment (4520×3120) ran in **186 s**, 3 360 tiles at 18.0 tiles/s. The prediction
shows papyrus weave texture but no legible letterforms in the sampled region.

There is a published team prediction for the same segment from a different model
(`new_canon_autoresearch_recipe-tile256-stride128`, 2.2 µm, 19080×13100). It is not a
useful reference: its output is mostly large blocky regions and isolated blobs, with no
letterforms either. Correlation between the two is 0.04, which says more about the
reference than about the pipeline. This segment is therefore **not** offered as a
correctness check — the w035 result above is the evidence.


### 2.4 µm coverage and its validation

Most scrolls do not publish a 9 µm render, but many publish a 2.4 µm render that pools
exactly into the representation the models were trained on. `--level 2 --z-pool 4`
reproduces `prepare_9um_isotropic_input` in memory: centre-crop Z to 84 planes (13..96),
average each group of four with round-half-to-even, giving 21 slices.

Validated on w035 against the native 9.36 µm render: same letters, **Pearson r = 0.73**
over a 1200×1200 region. That is the check that makes the 2.4 µm path trustworthy.

This roughly doubles usable coverage:

| representation | segments | scrolls |
| --- | --- | --- |
| native ≈9.4 µm | 106 | PHerc0139, PHerc0814, PHerc0841, PHerc0500P2 |
| 2.4 µm + level-2 pool | 196 | PHerc. Paris. 4 (80), PHerc0139 (38), PHerc1667 (38), PHerc0814 (19), PHerc0009B (18), PHerc0841 (3) |

It is also what makes PHerc1667 and PHerc. Paris. 4 reachable, since neither publishes a
9 µm render.


## Throughput

| workload | tiles | time |
| --- | --- | --- |
| 256×256 ROI | 9 | 0.5 s |
| PHerc0139 w035, full segment | 7 290 | 388 s |

Hardware: Intel Core Ultra 9 285K, 24 cores, 127 GB RAM, `torch 2.13.0+cpu`.

Downloading 854 MB of volume from S3 takes about two minutes with 32 workers and is cached.

## What it is not

- Not a replacement for the official tool: no test-time augmentation, no multi-GPU, no
  container, no native-resolution volumes.
- Not a new model. The value is access, not accuracy.
- The network definition is imported from a villa checkout; only the data path and
  preprocessing are reimplemented.

## Availability

Code: <!-- repo link -->
Licence: MIT. Model weights and data remain under their published terms.

Filed: November 2026 Progress Prize round, 2026-11-23. Repository state at filing is tagged `submission-nov-2026`.
