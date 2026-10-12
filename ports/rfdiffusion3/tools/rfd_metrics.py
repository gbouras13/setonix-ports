"""Summarise upstream's OWN per-design metrics across one or more output directories.

    python tools/rfd_metrics.py DIR [DIR ...] [--json FILE]

Every design RFdiffusion3 writes carries a sidecar `.json` with the engine's own measurements of the structure it just made
(`metrics`: `max_ca_deviation`, `n_chainbreaks`, the `n_clashing.*` counts, …). They are upstream's verdict on its own
output, computed by code this port does not touch, which makes them the most independent signal available here: a run whose
kernels are wrong shows it in these numbers before anything else does.

Per directory this prints the number of designs and, for every metric present, median and max across them.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Dict, List


def metrics_of(path: str) -> Dict[str, float]:
    try:
        d = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    m = d.get("metrics", d)
    return {k: float(v) for k, v in m.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    out = {}
    for d in a.dirs:
        files = sorted(f for f in os.listdir(d) if f.endswith(".json") and f != "opt_manifest.json") if os.path.isdir(d) else []
        rows: List[Dict[str, float]] = [m for m in (metrics_of(os.path.join(d, f)) for f in files) if m]
        if not rows:
            print(f"{d}: no per-design metrics")
            continue
        keys = sorted({k for r in rows for k in r})
        summary = {k: {"median": statistics.median([r[k] for r in rows if k in r]),
                       "max": max(r[k] for r in rows if k in r),
                       "n": sum(1 for r in rows if k in r)} for k in keys}
        out[d] = {"designs": len(rows), "metrics": summary}
        print(f"{d}: {len(rows)} designs")
        for k in keys:
            s = summary[k]
            print(f"    {k:<45} median {s['median']:>10.3f}   max {s['max']:>10.3f}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
