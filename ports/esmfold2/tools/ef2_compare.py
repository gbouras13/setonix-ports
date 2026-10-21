#!/usr/bin/env python3
"""Compare two ESMFold2 kit output directories prediction by prediction (the kit's file set, README 'Outputs').

    python ef2_compare.py A B [--quiet] [--json OUT]        exit 0 only if every prediction in A and B is byte-identical

Per prediction stem <id>__<fast|full>__s<seed>_x<k> (A and B must hold the same stems):
  cif   sha256 of the mmCIF text; when it differs: the _atom_site loop parsed, coordinates compared atom by atom (max |d| in A,
        CA RMSD without superposition, CA RMSD after Kabsch superposition), the B-factor column (upstream writes pLDDT there)
        compared, and the first differing column named
  npz   every array (pae, plddt, asym_id, mol_type, pair_chains_iptm, ...) exactly, else max |d|
  rows  pred_rows.jsonl: every field but the bookkeeping keys (pred_file gpu wall_s utc tag mode) -- iptm, ptm, n_tokens, msa_depths ...
Standard library + numpy only.
"""
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np

META = {"pred_file", "gpu", "wall_s", "utc", "tag", "mode"}
TOK = re.compile(r"'[^']*'|\"[^\"]*\"|\S+")


def sha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def atom_site(path):
    """The _atom_site loop as {column: [values]} (strings)."""
    lines = open(path).read().splitlines()
    cols, rows, i = [], [], 0
    while i < len(lines):
        if lines[i].startswith("_atom_site."):
            while i < len(lines) and lines[i].startswith("_atom_site."):
                cols.append(lines[i].split(".", 1)[1].strip())
                i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("_", "loop_", "#")):
                rows.append(TOK.findall(lines[i]))
                i += 1
            break
        i += 1
    return {c: [r[j] for r in rows] for j, c in enumerate(cols)}


def kabsch_rmsd(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    u, s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(((a @ r - b) ** 2).sum(1).mean()))


def cif_diff(pa, pb):
    A, B = atom_site(pa), atom_site(pb)
    out = {}
    if set(A) != set(B):
        out["columns"] = f"column sets differ: {sorted(set(A) ^ set(B))}"
        return out
    n = len(A.get("Cartn_x", []))
    if n != len(B.get("Cartn_x", [])):
        out["atoms"] = f"{n} vs {len(B.get('Cartn_x', []))}"
        return out
    xa = np.array([[float(A[c][k]) for c in ("Cartn_x", "Cartn_y", "Cartn_z")] for k in range(n)])
    xb = np.array([[float(B[c][k]) for c in ("Cartn_x", "Cartn_y", "Cartn_z")] for k in range(n)])
    d = np.abs(xa - xb).max(1)
    ca = np.array([nm.strip('"') == "CA" for nm in A["label_atom_id"]])
    out["atoms"] = n
    out["atoms_moved"] = int((d > 0).sum())
    out["max_abs_d_A"] = float(d.max()) if n else 0.0
    if ca.any():
        out["ca_rmsd_raw_A"] = float(np.sqrt(((xa[ca] - xb[ca]) ** 2).sum(1).mean()))
        out["ca_rmsd_kabsch_A"] = kabsch_rmsd(xa[ca], xb[ca])
    if "B_iso_or_equiv" in A:
        ba = np.array([float(v) for v in A["B_iso_or_equiv"]])
        bb = np.array([float(v) for v in B["B_iso_or_equiv"]])
        out["bfactor_max_abs_d"] = float(np.abs(ba - bb).max()) if n else 0.0
    others = [c for c in A if c not in ("Cartn_x", "Cartn_y", "Cartn_z", "B_iso_or_equiv") and A[c] != B[c]]
    if others:
        out["other_columns_differ"] = others
    return out


def npz_diff(pa, pb):
    a, b = np.load(pa), np.load(pb)
    out = {}
    for k in sorted(set(a.files) | set(b.files)):
        if k not in a.files or k not in b.files:
            out[k] = "one-sided"
            continue
        x, y = a[k], b[k]
        if x.shape != y.shape or x.dtype != y.dtype:
            out[k] = f"shape/dtype {x.shape}/{x.dtype} vs {y.shape}/{y.dtype}"
        elif not np.array_equal(x, y, equal_nan=True):
            out[k] = float(np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64))))
    return out


def rows(d):
    p = os.path.join(d, "pred_rows.jsonl")
    out = {}
    if os.path.isfile(p):
        for line in open(p):
            line = line.strip()
            if line:
                r = json.loads(line)
                stem = os.path.basename(r.get("pred_file", "")).rsplit(".cif", 1)[0]
                out[stem] = {k: v for k, v in r.items() if k not in META}
    return out


def main(argv):
    quiet = "--quiet" in argv
    js = argv[argv.index("--json") + 1] if "--json" in argv else None
    a, b = [x for x in argv if not x.startswith("--") and x != js][:2]
    ca = {os.path.basename(p)[:-4]: p for p in glob.glob(os.path.join(a, "cif_all", "*.cif"))}
    cb = {os.path.basename(p)[:-4]: p for p in glob.glob(os.path.join(b, "cif_all", "*.cif"))}
    ra, rb = rows(a), rows(b)
    stems = sorted(set(ca) | set(cb))
    report, n_same = [], 0
    for s in stems:
        rec = {"stem": s}
        if s not in ca or s not in cb:
            rec["status"] = "one-sided"
            report.append(rec)
            continue
        same_cif = sha(ca[s]) == sha(cb[s])
        na, nb = ca[s][:-4] + "_pae.npz", cb[s][:-4] + "_pae.npz"
        nd = npz_diff(na, nb) if os.path.isfile(na) and os.path.isfile(nb) else {"npz": "missing"}
        rd = {}
        if s in ra or s in rb:
            x, y = ra.get(s, {}), rb.get(s, {})
            rd = {k: (x.get(k), y.get(k)) for k in sorted(set(x) | set(y)) if x.get(k) != y.get(k)}
        rec["cif_identical"] = same_cif
        if not same_cif:
            rec["cif"] = cif_diff(ca[s], cb[s])
        if nd:
            rec["npz"] = nd
        if rd:
            rec["rows"] = rd
        rec["status"] = "identical" if (same_cif and not nd and not rd) else "differs"
        n_same += rec["status"] == "identical"
        report.append(rec)
    for rec in report:
        if quiet and rec["status"] == "identical":
            continue
        print(json.dumps(rec, sort_keys=True))
    print(f"COMPARE {a} vs {b}: {n_same}/{len(stems)} predictions byte-identical (cif + npz + rows)")
    if js:
        with open(js, "w") as fh:
            json.dump({"a": a, "b": b, "identical": n_same, "total": len(stems), "records": report}, fh, indent=1)
    return 0 if stems and n_same == len(stems) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
