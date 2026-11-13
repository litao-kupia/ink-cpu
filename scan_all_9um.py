"""Scan the open-data bucket for segments that already have a 9.36 um render."""

from __future__ import annotations

import concurrent.futures as futures
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

S3 = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_prefix(prefix: str) -> list[str]:
    for _ in range(3):
        try:
            r = requests.get(S3, params={"list-type": 2, "prefix": prefix,
                                         "delimiter": "/"}, timeout=60)
            if r.status_code == 200:
                root = ET.fromstring(r.text)
                return [p.find("s3:Prefix", NS).text
                        for p in root.findall("s3:CommonPrefixes", NS)]
        except Exception:
            pass
    return []


def segments_for(scroll: str) -> list[str]:
    return list_prefix(f"{scroll}/segments/")


def renders_for(seg_prefix: str) -> list[str]:
    vols = list_prefix(seg_prefix + "surface-volumes/")
    return [v for v in vols if "9.36" in v]


def main() -> int:
    samples = sorted(json.load(open("metadata.json"))["samples"])
    seg_lists = {}
    with futures.ThreadPoolExecutor(max_workers=8) as pool:
        for scroll, segs in zip(samples, pool.map(segments_for, samples)):
            seg_lists[scroll] = segs

    tasks = [(scroll, seg) for scroll, segs in seg_lists.items() for seg in segs]
    print(f"{len(tasks)} segments across {len(samples)} scrolls")

    hits: dict[str, list[str]] = {}
    with futures.ThreadPoolExecutor(max_workers=12) as pool:
        for (scroll, seg), vols in zip(tasks, pool.map(lambda t: renders_for(t[1]), tasks)):
            if vols:
                hits.setdefault(scroll, []).extend(vols)

    rows = []
    for scroll in samples:
        n = len(seg_lists[scroll])
        rows.append({"scroll": scroll, "segments": n,
                     "with_9um_render": len(hits.get(scroll, []))})

    Path("nine_um_ready.json").write_text(json.dumps(
        {"rows": rows, "paths": hits}, indent=1))

    print(f"\n{'scroll':16} {'segments':>9} {'9um ready':>10}")
    for r in sorted(rows, key=lambda r: -r["with_9um_render"]):
        if r["with_9um_render"] or r["segments"]:
            print(f"{r['scroll']:16} {r['segments']:9} {r['with_9um_render']:10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
