"""Compare two RFdiffusion3 output directories design by design.

    python tools/rfd_compare.py A_DIR B_DIR [--json FILE] [--quiet]

Each design is a gzipped mmCIF (`<spec>_<key>_<batch>_model_<k>.cif.gz`) plus a `.json` of upstream's own metadata. The
kit's `exact` class is stated as "byte-identical final coordinates and sequence indices", which is what this reads: the
`_atom_site` loop of each file, column by column — the written coordinates, the residue names (the designed sequence), the
atom and chain labels — and then the sidecar json.

Per design it prints one line: IDENTICAL, or what moved (atoms differing, max |dx| over the three axes in Angstrom, the
per-atom RMSD, and the first differing residue when the sequence itself changed). Files present in one directory only are
named. Exit 0 when every design matched, 1 otherwise — so a job can read the verdict from the exit code.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

ATOM_COLS = ("label_atom_id", "label_comp_id", "label_asym_id", "label_seq_id", "Cartn_x", "Cartn_y", "Cartn_z")


def read_atom_site(path: str) -> Tuple[List[dict], str]:
    """The `_atom_site` loop of a (gzipped) mmCIF as a list of dicts, plus the whole text."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    rows: List[dict] = []
    headers: List[str] = []
    in_loop = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("_atom_site."):
            headers.append(s.split(".", 1)[1].split()[0])
            in_loop = True
            continue
        if in_loop:
            if not s or s.startswith("#") or s.startswith("_") or s.startswith("loop_"):
                if rows:
                    break
                in_loop = bool(headers)
                continue
            parts = s.split()
            if len(parts) != len(headers):
                continue
            rows.append(dict(zip(headers, parts)))
    return rows, text


def compare_one(a: str, b: str) -> dict:
    ra, ta = read_atom_site(a)
    rb, tb = read_atom_site(b)
    out: Dict[str, object] = {"file": os.path.basename(a), "atoms_a": len(ra), "atoms_b": len(rb),
                              "text_identical": ta == tb}
    if len(ra) != len(rb):
        out["verdict"] = f"ATOM COUNT {len(ra)} vs {len(rb)}"
        return out
    ndiff = 0
    maxd = 0.0
    sq = 0.0
    seq_diffs: List[str] = []
    for i, (x, y) in enumerate(zip(ra, rb)):
        moved = False
        d2 = 0.0
        for c in ("Cartn_x", "Cartn_y", "Cartn_z"):
            if x.get(c) != y.get(c):
                moved = True
            try:
                d = float(x.get(c, "nan")) - float(y.get(c, "nan"))
            except ValueError:
                d = float("nan")
            d2 += d * d
            if not math.isnan(d):
                maxd = max(maxd, abs(d))
        sq += d2
        if moved:
            ndiff += 1
        for c in ("label_comp_id", "label_asym_id", "label_seq_id", "label_atom_id"):
            if x.get(c) != y.get(c) and len(seq_diffs) < 5:
                seq_diffs.append(f"atom {i} {c}: {x.get(c)} vs {y.get(c)}")
    out.update({"atoms_differing": ndiff, "max_abs_delta_A": maxd,
                "rmsd_A": math.sqrt(sq / len(ra)) if ra else 0.0, "label_diffs": seq_diffs})
    out["verdict"] = ("IDENTICAL" if ndiff == 0 and not seq_diffs else
                      f"{ndiff}/{len(ra)} atoms moved, max |d| {maxd:.3f} A, rmsd {out['rmsd_A']:.3f} A"
                      + (f"; labels differ ({seq_diffs[0]})" if seq_diffs else ""))
    return out


def compare_json(a: str, b: str) -> Optional[str]:
    try:
        da, db = json.load(open(a, encoding="utf-8")), json.load(open(b, encoding="utf-8"))
    except (OSError, ValueError) as e:
        return f"json unreadable ({type(e).__name__})"
    if da == db:
        return None
    keys = sorted(set(da) | set(db))
    diffs = [f"{k}: {da.get(k)!r} vs {db.get(k)!r}" for k in keys if da.get(k) != db.get(k)]
    return "; ".join(diffs[:4]) + (f" (+{len(diffs) - 4} more)" if len(diffs) > 4 else "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--json", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    def designs(d: str) -> Dict[str, str]:
        return {f: os.path.join(d, f) for f in sorted(os.listdir(d))
                if f.endswith(".cif.gz") and "_denoised_" not in f and "_noisy_" not in f} if os.path.isdir(d) else {}

    A, B = designs(args.a), designs(args.b)
    only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
    both = sorted(set(A) & set(B))
    results, ok = [], True
    for f in both:
        r = compare_one(A[f], B[f])
        jd = compare_json(A[f][:-7] + ".json", B[f][:-7] + ".json") if os.path.isfile(A[f][:-7] + ".json") else None
        r["json_diff"] = jd
        results.append(r)
        same = r["verdict"] == "IDENTICAL"
        ok = ok and same
        if not args.quiet or not same:
            print(f"{f}: {r['verdict']}" + (f" | json: {jd}" if jd else ""))
    for f in only_a:
        print(f"ONLY IN A: {f}"); ok = False
    for f in only_b:
        print(f"ONLY IN B: {f}"); ok = False
    by_key = {}
    for r in results:                                  # <spec>_<key>_<batch>_model_<k>.cif.gz
        parts = r["file"].rsplit("_model_", 1)[0].rsplit("_", 1)[0]
        by_key.setdefault(parts, [0, 0])
        by_key[parts][1] += 1
        by_key[parts][0] += int(r["verdict"] == "IDENTICAL")
    n_same = sum(1 for r in results if r["verdict"] == "IDENTICAL")
    comparable = [r for r in results if "max_abs_delta_A" in r]
    n_count = len(results) - len(comparable)
    worst = max((r["max_abs_delta_A"] for r in comparable), default=0.0)
    print(f"\n{n_same}/{len(results)} designs identical; "
          + (f"{n_count} with a different ATOM COUNT (not comparable atom by atom); " if n_count else "")
          + (f"worst max |d| {worst:.3f} A over the {len(comparable)} comparable" if comparable else "none comparable")
          + ("" if not (only_a or only_b) else f"; {len(only_a)} only in A, {len(only_b)} only in B"))
    if len(by_key) > 1:
        print("    per key: " + "  ".join(f"{k}: {v[0]}/{v[1]}" for k, v in sorted(by_key.items())))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"a": args.a, "b": args.b, "identical": n_same, "designs": len(results), "per_key": by_key,
                       "worst_max_abs_delta_A": worst, "only_a": only_a, "only_b": only_b, "results": results}, fh, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
