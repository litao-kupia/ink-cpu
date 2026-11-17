# November 2026 Progress Prize — draft form answers

Paste these into the form at
https://docs.google.com/forms/d/e/1FAIpQLSeB6MC_--4UCCEGc3zalX57-uE4puRcE_vJoYJCrEt4uYaaaw/viewform
(deadline Nov 30, 11:59pm Pacific). Adjust name / team / Discord handle.

## URL of the open source / publicly available contribution

<repo URL>

## What is your contribution?

**(1) Which scroll data did you work on?**

PHerc0139 `w035` as a validation segment (it is in the 9 µm models' training set, so
correct output is known); PHerc0500P2 as a fragment control with a published team
prediction; and **PHerc0009B (18 segments) and PHerc0841 (3 segments)** as unread scrolls.
The coverage survey spans all 328 traced segments across 45 scrolls in the open-data
catalog.

**(2) How does it substantially increase the probability of yourself or someone else
reading those scrolls or others?**

It removes the hardware and build gate from the 9 µm First Letters workflow. That workflow
previously required Linux, an NVIDIA GPU, and building `vc_reader` (a C++ extension) from
source; the official tutorial says as much. It now runs on a desktop CPU in ~200 lines of
Python with no build step, at 18 tiles/s, full segment in 6.5 minutes.

Two concrete consequences:

- **Screening a whole scroll becomes affordable.** I screened all 18 segments of
  PHerc0009B — 121,383 tiles, 115 minutes — on one desktop. Deciding whether a scroll is
  worth a labelling effort is now a two-hour question instead of a GPU project.
- **Coverage roughly doubles.** Adding the 2.4 µm → 9 µm level-2 pooling path raises usable
  segments from 106 to 196 and brings PHerc. Paris. 4 (80 segments) and PHerc1667 (38) into
  reach, neither of which publishes a 9 µm render.

**(3) What does it enable that was not possible before?**

Running the shared cross-scroll ink models on ordinary hardware, against published surface
volumes, as a single command — and therefore screening a scroll end to end before
committing to pseudo-labelling. It also quantifies the cross-scroll generalisation
bottleneck: on 21 segments across two unread scrolls the released model produces structure
but no strokes, while rendering text cleanly on a scroll it trained on (w035).

**(4) What evidence have you provided?**

- **Parity check:** the 2.4 µm pooled path against the native 9.37 µm render on w035 —
  same letters, Pearson r = 0.73 over a 1200×1200 region.
- **Correctness control:** legible, correctly oriented Greek letterforms on w035, full
  5820×5240 segment, 7,290 tiles, 388 s, no tiling seams.
- **Real use:** 21 segments across two unread scrolls, with per-segment scores
  (`frac>0.6`, p99) and the negative result recorded, including an ensemble + depth-offset
  re-run of the best segment.
- **Coverage survey:** resolution-by-segment scan of all 328 traced segments in 45 scrolls,
  published as data alongside the tool.
- **Throughput:** measured on a 24-core desktop with no GPU, in the README.

## Honest framing for the reviewer

The tool did not find text. What it did was make the search cheap enough to run, validate
that the cheap path agrees with the expensive one (r = 0.73), and report a clean negative
on 21 unread segments rather than a hopeful one.
