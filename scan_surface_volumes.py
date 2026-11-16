"""Count published surface volumes per segment, bucketed by resolution."""

from __future__ import annotations

import concurrent.futures as futures
import json
import re
import xml.etree.ElementTree as ET

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


def res_of(path: str) -> str:
    m = re.search(r"/(\d+\.?\d*)um-", path)
    return f"{float(m.group(1)):.1f}" if m else "?"


def main() -> int:
    samples = sorted(json.load(open("metadata.json"))["samples"])
    with futures.ThreadPoolExecutor(max_workers=8) as pool:
        seg_lists = dict(zip(samples, pool.map(
            lambda s: list_prefix(f"{s}/segments/"), samples)))

    tasks = [(scroll, seg) for scroll, segs in seg_lists.items() for seg in segs]

    def scan(t):
        scroll, seg = t
        return t, list_prefix(seg + "surface-volumes/")

    table = {}
    with futures.ThreadPoolExecutor(max_workers=12) as pool:
        for (scroll, seg), vols in pool.map(scan, tasks):
            if vols:
                table[seg] = vols

    json.dump(table, open("surface_volumes.json", "w"), indent=1)

    by_scroll: dict[str, dict[str, int]] = {}
    for seg, vols in table.items():
        scroll = seg.split("/")[0]
        for v in vols:
            r = res_of(v)
            by_scroll.setdefault(scroll, {}).setdefault(r, 0)
            by_scroll[scroll][r] += 1

    res_all = sorted({r for d in by_scroll.values() for r in d})
    header = f"{'scroll':16} {'segs':>5} " + " ".join(f"{r:>7}" for r in res_all)
    print(header)
    for scroll in samples:
        d = by_scroll.get(scroll, {})
        if not d and not seg_lists[scroll]:
            continue
        print(f"{scroll:16} {len(seg_lists[scroll]):5} " +
              " ".join(f"{d.get(r, 0):7}" for r in res_all))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
