"""Are these designs proteins at all? A cheap, dependency-free read of RFdiffusion3's output structures.

    python tools/rfd_sanity.py OUT_DIR [OUT_DIR ...] [--json FILE]

A run can exit 0 and still have produced noise — a wrong normalisation kernel or a mis-replayed graph does exactly that. Per
design this reads the mmCIF `_atom_site` loop and reports what a backbone must satisfy:

  CA-CA      consecutive alpha-carbon distance within a chain: 3.80 +- 0.05 A in a real chain (cis-proline aside)
  Rg         radius of gyration against the compact-globule expectation 2.2 * N^0.38 A — a random cloud is far above it
  clash      the closest CA-CA pair that is not a neighbour; below ~3.5 A is a clash
  finite     no NaN / inf coordinate

It prints one line per design and a PASS/FAIL per directory. This is a floor, not a quality measure: passing says the model
produced protein-like backbones, not that they are good designs.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
from typing import Dict, List, Tuple


def ca_atoms(path: str) -> Tuple[Dict[str, List[Tuple[float, float, float]]], int, bool]:
    """{chain: [CA xyz ...]}, total atom count, all-finite."""
    opener = gzip.open if path.endswith(".gz") else open
    headers: List[str] = []
    chains: Dict[str, List[Tuple[float, float, float]]] = {}
    n_atoms = 0
    finite = True
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        in_loop = False
        for line in fh:
            s = line.strip()
            if s.startswith("_atom_site."):
                headers.append(s.split(".", 1)[1].split()[0])
                in_loop = True
                continue
            if not in_loop:
                continue
            if not s or s.startswith("#") or s.startswith("loop_") or s.startswith("_"):
                if n_atoms:
                    break
                continue
            p = s.split()
            if len(p) != len(headers):
                continue
            row = dict(zip(headers, p))
            n_atoms += 1
            try:
                xyz = tuple(float(row[c]) for c in ("Cartn_x", "Cartn_y", "Cartn_z"))
            except (KeyError, ValueError):
                finite = False
                continue
            if not all(math.isfinite(v) for v in xyz):
                finite = False
                continue
            if row.get("label_atom_id") == "CA":
                chains.setdefault(row.get("label_asym_id", "?"), []).append(xyz)
    return chains, n_atoms, finite


def d(a, b) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def judge(path: str) -> dict:
    chains, n_atoms, finite = ca_atoms(path)
    cas = [p for c in chains.values() for p in c]
    n = len(cas)
    out = {"file": os.path.basename(path), "atoms": n_atoms, "chains": len(chains), "ca": n, "finite": finite}
    if n < 3:
        out["verdict"] = "FAIL: fewer than 3 CA atoms"
        return out
    steps = [d(c[i], c[i + 1]) for c in chains.values() for i in range(len(c) - 1)]
    cen = [sum(p[i] for p in cas) / n for i in range(3)]
    rg = math.sqrt(sum(d(p, cen) ** 2 for p in cas) / n)
    rg_expect = 2.2 * n ** 0.38
    worst = min((d(cas[i], cas[j]) for i in range(n) for j in range(i + 2, n)), default=float("inf"))
    bad_steps = sum(1 for s in steps if not (3.6 <= s <= 4.0))
    out.update({"ca_ca_min": min(steps), "ca_ca_max": max(steps),
                "ca_ca_mean": sum(steps) / len(steps), "ca_ca_outliers": bad_steps,
                "rg": rg, "rg_expected": rg_expect, "rg_ratio": rg / rg_expect, "closest_nonneighbour_ca": worst})
    fails = []
    if not finite:
        fails.append("non-finite coordinates")
    if bad_steps > 0.02 * max(1, len(steps)):
        fails.append(f"{bad_steps}/{len(steps)} CA-CA steps outside 3.6-4.0 A")
    if rg / rg_expect > 2.0:
        fails.append(f"Rg {rg:.1f} A is {rg / rg_expect:.1f}x the compact expectation")
    if worst < 3.0:
        fails.append(f"CA clash at {worst:.2f} A")
    out["verdict"] = "ok" if not fails else "FAIL: " + "; ".join(fails)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--json", default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    allres, ok_all = {}, True
    for dd in a.dirs:
        files = sorted(f for f in os.listdir(dd) if f.endswith(".cif.gz")
                       and "_denoised_" not in f and "_noisy_" not in f) if os.path.isdir(dd) else []
        res = [judge(os.path.join(dd, f)) for f in files]
        allres[dd] = res
        bad = [r for r in res if r["verdict"] != "ok"]
        ok_all = ok_all and files and not bad
        for r in res:
            if not a.quiet or r["verdict"] != "ok":
                print(f"{r['file']}: {r['verdict']}" + (
                    "" if r["verdict"] != "ok" else
                    f" (CA {r['ca']}, CA-CA {r['ca_ca_mean']:.2f} A, Rg {r['rg']:.1f}/{r['rg_expected']:.1f} A,"
                    f" closest {r['closest_nonneighbour_ca']:.2f} A)"))
        print(f"{dd}: {len(res) - len(bad)}/{len(res)} designs protein-like"
              + ("" if files else "  (NO DESIGNS)"))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(allres, fh, indent=1)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
