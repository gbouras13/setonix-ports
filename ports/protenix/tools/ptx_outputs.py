#!/usr/bin/env python3
"""Summarise and compare Protenix prediction outputs (stock's layout, which the kit's routes keep).

  ptx_outputs.py summary <dir> [<dir> ...]                   every *_sample_<k>.cif under each dir: atoms, CA count, non-finite
                                                             coordinates, and its summary confidence JSON's scores
  ptx_outputs.py compare [--identical] <ref_dir> <dir> [...] per sample present in the reference (matched by its path below the dir):
                                                             byte identity of the CIF and of its confidence JSON, CA RMSD after Kabsch
                                                             superposition and the raw (unaligned) max CA deviation, score deltas

Layout: <out>/<name>/seed_<seed>/predictions/<name>_sample_<k>.cif with <name>_summary_confidence_sample_<k>.json beside it (<k> is the
rank by confidence, so only the same path is a valid pair: the samples of one run differ by design). Columns are located from the
_atom_site. loop header, never assumed. Engineering check, not a structural claim. Exit status: 0; 1 if a sample is missing from a
compared dir, or any coordinate or score is not finite (NaN is the gfx90a failure mode of PTX inline-asm kernels: job 49905402); with
--identical, also 1 unless every CIF and JSON is byte-identical to the reference's.
"""
import glob
import hashlib
import json
import math
import os
import re
import shlex
import sys

import numpy as np

SCORES = ("plddt", "ptm", "iptm", "ranking_score")


def samples(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "**", "*_sample_*.cif"), recursive=True)):
        if re.search(r"_sample_\d+\.cif$", p) and "_summary_confidence_" not in p:
            out[os.path.relpath(p, d)] = p
    return out


def conf_path(cif):
    return re.sub(r"_sample_(\d+)\.cif$", r"_summary_confidence_sample_\1.json", cif)


def atoms(path):
    cols, rows, in_loop, started = [], [], False, False
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("_atom_site."):
                cols.append(s.split(".", 1)[1]); in_loop = True; continue
            if in_loop and cols:
                if not s or s.startswith("#") or s.startswith("loop_") or s.startswith("_"):
                    if started:
                        break
                    continue
                started = True
                parts = shlex.split(s, posix=True) if ('"' in s or "'" in s) else s.split()
                if len(parts) == len(cols):
                    rows.append(parts)
    ix = {c: i for i, c in enumerate(cols)}

    def col(*names):
        for n in names:
            if n in ix:
                return ix[n]
        return None
    ia, ich, ires = col("label_atom_id", "auth_atom_id"), col("label_asym_id", "auth_asym_id"), col("label_seq_id", "auth_seq_id")
    ix_, iy, iz = col("Cartn_x"), col("Cartn_y"), col("Cartn_z")
    xyz, ca = [], {}
    for r in rows:
        try:
            p = (float(r[ix_]), float(r[iy]), float(r[iz]))
        except (TypeError, ValueError, IndexError):
            p = (math.nan, math.nan, math.nan)
        xyz.append(p)
        if ia is not None and r[ia] == "CA":
            ca[(r[ich] if ich is not None else "?", r[ires] if ires is not None else "?")] = p
    return np.array(xyz, dtype=float).reshape(-1, 3), ca


def scores(cif):
    j = conf_path(cif)
    if not os.path.isfile(j):
        return {"json": "absent"}
    try:
        c = json.load(open(j))
    except Exception as e:  # noqa: BLE001
        return {"json_error": repr(e)}
    return {k: c[k] for k in SCORES if isinstance(c.get(k), (int, float))}


def finite(sc):
    return "json_error" not in sc and all(math.isfinite(v) for k, v in sc.items() if k in SCORES)


def kabsch_rmsd(a, b):
    a0, b0 = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(a0.T @ b0)
    dd = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1.0, 1.0, dd]) @ vt
    return float(np.sqrt(((a0 @ r - b0) ** 2).sum(1).mean()))


def sha(p):
    if not os.path.isfile(p):
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fmt(sc):
    return " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in sc.items())


def summary(dirs):
    bad = 0
    for d in dirs:
        ss = samples(d)
        print(f"== {d}: {len(ss)} sample file(s)")
        if not ss:
            bad += 1
        for rel, p in ss.items():
            xyz, ca = atoms(p)
            nonfin = int((~np.isfinite(xyz)).sum())
            sc = scores(p)
            ok = nonfin == 0 and len(xyz) > 0 and finite(sc)
            bad += 0 if ok else 1
            print(f"  {rel}: atoms={len(xyz)} CA={len(ca)} nonfinite_xyz={nonfin} {fmt(sc)}" + ("" if ok else "  <-- NOT FINITE / EMPTY"))
    return 1 if bad else 0


def compare(ref, others, identical=False):
    bad, rs = 0, samples(ref)
    if not rs:
        print(f"no *_sample_<k>.cif under {ref}")
        return 1
    for o in others:
        os_ = samples(o)
        print(f"== {o} vs {ref}")
        n_same = 0
        for rel, rp in rs.items():
            op = os_.get(rel)
            if op is None:
                print(f"  {rel}: MISSING in {o}"); bad += 1; continue
            same_cif, same_json = sha(rp) == sha(op), sha(conf_path(rp)) == sha(conf_path(op))
            n_same += same_cif and same_json
            xa, ca_a = atoms(rp)
            xb, ca_b = atoms(op)
            keys = [k for k in ca_a if k in ca_b]
            A = np.array([ca_a[k] for k in keys]).reshape(-1, 3)
            B = np.array([ca_b[k] for k in keys]).reshape(-1, 3)
            if len(keys) >= 3 and np.isfinite(A).all() and np.isfinite(B).all():
                rmsd, raw = kabsch_rmsd(A, B), float(np.sqrt(((A - B) ** 2).sum(1)).max())
            else:
                rmsd = raw = math.nan
            sa, sb = scores(rp), scores(op)
            if not (np.isfinite(xb).all() and finite(sb)) or math.isnan(rmsd):
                bad += 1
            if identical and not (same_cif and same_json):
                bad += 1
            deltas = " ".join(f"d{k}={sb[k] - sa[k]:+.4f}" for k in SCORES if k in sa and k in sb)
            print(f"  {rel}: bytes_identical={same_cif and same_json} (cif {same_cif}, json {same_json}) CA_shared={len(keys)} "
                  f"CA_RMSD={rmsd:.3f}A raw_max={raw:.3f}A {deltas}")
        extra = sorted(set(os_) - set(rs))
        if extra:
            print(f"  not in the reference: {', '.join(extra)}")
        print(f"  {n_same}/{len(rs)} sample(s) byte-identical")
    return 1 if bad else 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    ident = "--identical" in argv
    argv = [a for a in argv if a != "--identical"]
    if len(argv) < 2 or argv[0] not in ("summary", "compare") or (argv[0] == "compare" and len(argv) < 3):
        print(__doc__)
        sys.exit(2)
    sys.exit(summary(argv[1:]) if argv[0] == "summary" else compare(argv[1], argv[2:], ident))
