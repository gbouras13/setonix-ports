#!/usr/bin/env python3
"""Compare deterministic-mode 1BRS runs bit for bit (valid_det.sbatch).

  compare_det.py <OUT> A:B [C:D ...]

Each arm <OUT>/<label>/ holds the prediction dump(s) the setonix_det hook wrote (dump/pred_r<rank>_0.pt: coordinate, plddt / pae / pde /
resolved logits, contact_probs, summary) and the CLI's own outputs (out/**/summary_confidence_sample_<k>.json, *_sample_<k>.cif). Per
pair: every dumped tensor bitwise (else max abs and relative L2), per-sample coordinate RMSD (raw, and after superposition), and the
summary confidences. An arm without a dump is compared on its JSON confidences and CIF coordinates (printed to 3 decimals) instead.
A TP arm's rank dumps are also compared with each other.
"""
import glob
import json
import os
import re
import sys

import numpy as np
import torch

OUT = sys.argv[1]
PAIRS = [p.split(":") for p in sys.argv[2:]]      # a side may be label@<other OUT dir> (a run from another job)


def where(side):
    label, _, d = side.partition("@")
    return label, (d or OUT)
KEYS = ("coordinate", "plddt", "pae", "pde", "resolved", "contact_probs")
SUM = ("plddt", "ptm", "iptm", "ranking_score")


def load_dump(side, rank=0):
    label, base = where(side)
    p = sorted(glob.glob(os.path.join(base, label, "dump", f"pred_r{rank}_*.pt")))
    return torch.load(p[0], weights_only=False) if p else None


def load_cli(side):
    label, base = where(side)
    OUTD = base
    js = sorted(glob.glob(os.path.join(OUTD, label, "out", "**", "*_summary_confidence_sample_*.json"), recursive=True))
    conf = {int(re.search(r"_sample_(\d+)\.json$", j).group(1)): json.load(open(j)) for j in js}
    xyz = {}
    for c in sorted(glob.glob(os.path.join(OUTD, label, "out", "**", "*_sample_*.cif"), recursive=True)):
        if "summary" in c:
            continue
        k = int(re.search(r"_sample_(\d+)\.cif$", c).group(1))
        cols, rows, in_loop = [], [], False
        for line in open(c):
            if line.startswith("_atom_site."):
                cols.append(line.strip().split(".", 1)[1]); in_loop = True; continue
            if in_loop and line.startswith(("ATOM", "HETATM")):
                d = dict(zip(cols, line.split()))
                rows.append([float(d["Cartn_x"]), float(d["Cartn_y"]), float(d["Cartn_z"])])
            elif in_loop and cols and line.strip() == "#":
                in_loop = False
        xyz[k] = np.array(rows)
    return conf, xyz


def kabsch(a, b):
    a0, b0 = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(a0.T @ b0)
    d = np.sign(np.linalg.det(u @ vt))
    return float(np.sqrt(((a0 @ (u @ np.diag([1.0, 1.0, d]) @ vt) - b0) ** 2).sum(1).mean()))


def tdiff(a, b):
    if a.shape != b.shape:
        return f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    if a.dtype == b.dtype and torch.equal(a, b):
        return "="
    d = a.double() - b.double()
    return f"max|d| {float(d.abs().max()):.2e} relL2 {float(d.norm()) / (float(b.double().norm()) or 1.0):.2e}"


def sval(s, k):
    v = s.get(k)
    return float(v) if torch.is_tensor(v) or isinstance(v, (int, float)) else None


print(f"==================== DETERMINISTIC 1BRS: stock vs kit ({OUT}) ====================")
labels = sorted({x for p in PAIRS for x in p})
for L in labels:
    d = load_dump(L)
    r1 = load_dump(L, 1)
    conf, _ = load_cli(L)
    det = d.get("deterministic") if d else None
    means = {k: np.mean([c[k] for c in conf.values()]) for k in SUM if conf and all(k in c for c in conf.values())}
    print(f"  {L}: dump {'yes' if d else 'NO'} (deterministic={det}){'  rank-1 dump yes' if r1 else ''}  samples {len(conf)}  "
          + " ".join(f"{k} {v:.4f}" for k, v in means.items()))
    if d and r1:
        print(f"      rank 0 vs rank 1: " + "  ".join(f"{k}:{tdiff(d[k], r1[k])}" for k in KEYS if k in d and k in r1))
verdict = {}
for a, b in PAIRS:
    print(f"\n{a} vs {b}")
    da, db = load_dump(a), load_dump(b)
    if da and db:
        cells = {k: tdiff(da[k], db[k]) for k in KEYS if k in da and k in db}
        exact = all(v == "=" for v in cells.values())
        print("  tensors: " + "  ".join(f"{k}:{v}" for k, v in cells.items()))
        ca, cb = da["coordinate"].double().numpy(), db["coordinate"].double().numpy()
        raw = [float(np.sqrt(((ca[i] - cb[i]) ** 2).sum(-1).mean())) for i in range(ca.shape[0])]
        sup = [kabsch(ca[i], cb[i]) for i in range(ca.shape[0])]
        print("  coordinate RMSD per sample, raw (A): " + " ".join(f"{x:.2e}" for x in raw) + "   superposed: " + " ".join(f"{x:.2e}" for x in sup))
        sa, sb = da.get("summary", []), db.get("summary", [])
        rows = []
        for i, (x, y) in enumerate(zip(sa, sb)):
            rows.append(" ".join(f"{k} {sval(x, k):.4f}/{sval(y, k):.4f}" for k in SUM if sval(x, k) is not None and sval(y, k) is not None))
        for i, r in enumerate(rows):
            print(f"  sample {i}: {r}")
        verdict[f"{a}:{b}"] = "BITWISE" if exact else f"differs (coord raw RMSD max {max(raw):.2e} A)"
    else:
        ca_, xa = load_cli(a)
        cb_, xb = load_cli(b)
        ks = sorted(set(xa) & set(xb))
        if not ks or not ca_ or not cb_:                    # a side with no outputs is MISSING, never "identical"
            print(f"  MISSING outputs: {a} has {len(ca_)} confidence / {len(xa)} CIF files, {b} has {len(cb_)} / {len(xb)}")
            verdict[f"{a}:{b}"] = "MISSING"
            continue
        same_json = all(ca_.get(k) == cb_.get(k) for k in ks)
        raw = [float(np.sqrt(((xa[k] - xb[k]) ** 2).sum(-1).mean())) if xa[k].shape == xb[k].shape else float("nan") for k in ks]
        print(f"  (no dump for {'both' if not (da or db) else (a if not da else b)}: CLI outputs) confidence JSONs identical: {same_json}  "
              f"CIF coordinate RMSD per sample (3-decimal print): " + " ".join(f"{x:.3f}" for x in raw))
        for k in ks:
            print(f"  sample {k}: " + " ".join(f"{m} {ca_[k].get(m, float('nan')):.4f}/{cb_[k].get(m, float('nan')):.4f}" for m in SUM))
        verdict[f"{a}:{b}"] = "CLI-identical" if same_json and all(x == 0 for x in raw) else f"differs (CIF RMSD max {max(raw) if raw else float('nan'):.3f} A)"
print("\nVDET " + "  ".join(f"{k}={v}" for k, v in verdict.items()))
