#!/usr/bin/env python3
"""Summarise and compare Boltz-2 prediction outputs (stock `boltz predict` layout or the kit's by_seed/ layout).

  b2s_outputs.py summary <dir> [<dir> ...]      every *_model_<k>.cif found under each dir: atoms, CA count, non-finite
                                                coordinates, the confidence JSON's scores, the pLDDT npz's finiteness
  b2s_outputs.py compare <ref_dir> <dir> [...]  per model file present in both (matched by basename): byte identity, CA RMSD
                                                after Kabsch superposition and raw (unaligned) max CA deviation, score deltas

Layouts: stock writes <out>/boltz_results_<name>/predictions/<name>/<name>_model_<k>.cif with confidence_<name>_model_<k>.json and
plddt_<name>_model_<k>.npz beside it; the kit writes the same file set under <out>/by_seed/<name>/s<seed>/. Columns are located from
the _atom_site. loop header, never assumed. Engineering check, not a structural claim. Exit status: 0, or 1 if any model has a
non-finite coordinate or score (NaN is the gfx90a failure mode of PTX inline-asm kernels: Protenix port, 2026-09-24).
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


def models(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "**", "*_model_*.cif"), recursive=True)):
        out.setdefault(os.path.basename(p), p)
    return out


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
        except (TypeError, ValueError):
            p = (math.nan, math.nan, math.nan)
        xyz.append(p)
        if ia is not None and r[ia] == "CA":
            ca[(r[ich] if ich is not None else "?", r[ires] if ires is not None else "?")] = p
    return np.array(xyz, dtype=float).reshape(-1, 3), ca


def scores(cif_path):
    d, b = os.path.dirname(cif_path), os.path.basename(cif_path)
    stem = b[:-4]                                    # <name>_model_<k>
    out = {}
    for j in (os.path.join(d, f"confidence_{stem}.json"),):
        if os.path.isfile(j):
            try:
                c = json.load(open(j))
                for k in ("confidence_score", "ptm", "iptm", "complex_plddt", "complex_iplddt", "complex_pde"):
                    if k in c:
                        out[k] = c[k]
            except Exception as e:  # noqa: BLE001
                out["json_error"] = repr(e)
    npz = os.path.join(d, f"plddt_{stem}.npz")
    if os.path.isfile(npz):
        a = np.load(npz)
        arr = a[a.files[0]]
        out["plddt_npz_n"] = int(arr.size)
        out["plddt_npz_nonfinite"] = int((~np.isfinite(arr)).sum())
        out["plddt_npz_mean"] = float(np.nanmean(arr)) if arr.size else None
    return out


def kabsch_rmsd(a, b):
    a0, b0 = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(a0.T @ b0)
    dd = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1.0, 1.0, dd]) @ vt
    return float(np.sqrt(((a0 @ r - b0) ** 2).sum(1).mean()))


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def finite_scores(sc):
    return all(isinstance(v, (int, float)) and math.isfinite(v) for k, v in sc.items() if k not in ("json_error", "plddt_npz_nonfinite", "plddt_npz_n")) \
        and not sc.get("plddt_npz_nonfinite")


def summary(dirs):
    bad = 0
    for d in dirs:
        ms = models(d)
        print(f"== {d}: {len(ms)} model file(s)")
        for b, p in ms.items():
            xyz, ca = atoms(p)
            nonfin = int((~np.isfinite(xyz)).sum())
            sc = scores(p)
            ok = nonfin == 0 and len(xyz) > 0 and finite_scores(sc)
            bad += 0 if ok else 1
            print(f"  {b}: atoms={len(xyz)} CA={len(ca)} nonfinite_xyz={nonfin} " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in sc.items()) + ("" if ok else "  <-- NOT FINITE / EMPTY"))
    return 1 if bad else 0


def compare(ref, others):
    bad = 0
    rm = models(ref)
    for o in others:
        om = models(o)
        print(f"== {o} vs {ref}")
        common = sorted(set(rm) & set(om))
        if not common:
            print("  no model file in common (by basename)"); bad += 1; continue
        for b in common:
            ident = sha(rm[b]) == sha(om[b])
            xa, ca_a = atoms(rm[b]); xb, ca_b = atoms(om[b])
            keys = [k for k in ca_a if k in ca_b]
            A = np.array([ca_a[k] for k in keys]); B = np.array([ca_b[k] for k in keys])
            if len(keys) >= 3 and np.isfinite(A).all() and np.isfinite(B).all():
                rmsd = kabsch_rmsd(A, B); raw = float(np.sqrt(((A - B) ** 2).sum(1)).max())
            else:
                rmsd = raw = math.nan; bad += 1
            sa, sb = scores(rm[b]), scores(om[b])
            deltas = " ".join(f"d{k}={sb[k] - sa[k]:+.4f}" for k in ("confidence_score", "ptm", "iptm", "complex_plddt") if isinstance(sa.get(k), float) and isinstance(sb.get(k), float))
            print(f"  {b}: bytes_identical={ident} CA_shared={len(keys)} CA_RMSD={rmsd:.3f}A raw_max={raw:.3f}A {deltas}")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("summary", "compare"):
        print(__doc__); sys.exit(2)
    sys.exit(summary(sys.argv[2:]) if sys.argv[1] == "summary" else compare(sys.argv[2], sys.argv[3:]))
