#!/usr/bin/env python3
"""compare_structs.py MODEL REFERENCE [--chain-map A:A] [--model-chains A] [--ref-chains A] [--resmin N] [--resmax M]

CA-level comparison of two structures, for judging a GPU prediction against (a) the same
stack's CPU prediction and (b) an experimental structure.

Residues are paired by (chain, author residue number) and a pair is kept only when the
residue NAMES also agree -- so a numbering offset shows up as "0 paired", never as a
silently wrong RMSD. Prints: paired count, identity check, CA RMSD after optimal (Kabsch)
superposition, GDT_TS-style fractions under 1/2/4/8 A, and mean CA pLDDT (B-factor) of
each file over the paired residues.
"""
import argparse
import sys

import gemmi
import numpy as np


def ca_table(path, chain_map, lo, hi, keep=None):
    st = gemmi.read_structure(path)
    st.setup_entities()
    out = {}
    for ch in st[0]:
        if keep and ch.name not in keep:          # compare ONLY these chains of this file (a multi-chain
            continue                              # reference would otherwise pair every same-named chain)
        cname = chain_map.get(ch.name, ch.name) if chain_map else ch.name
        for r in ch:
            if r.het_flag == "H":
                continue
            n = r.seqid.num
            if (lo is not None and n < lo) or (hi is not None and n > hi):
                continue
            ca = r.find_atom("CA", "*")
            if ca is None:
                continue
            out[(cname, n)] = (r.name, np.array(ca.pos.tolist()), ca.b_iso)
    return out


def kabsch_rmsd(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt
    diff = Pc @ R - Qc
    per = np.sqrt((diff ** 2).sum(1))
    return float(np.sqrt((per ** 2).mean())), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("ref")
    ap.add_argument("--chain-map", default="", help="model->ref chain renames, e.g. A:B,B:C")
    ap.add_argument("--model-chains", default="", help="keep only these model chains, e.g. A or A,B")
    ap.add_argument("--ref-chains", default="", help="keep only these reference chains")
    ap.add_argument("--resmin", type=int)
    ap.add_argument("--resmax", type=int)
    a = ap.parse_args()
    cmap = dict(kv.split(":") for kv in a.chain_map.split(",") if kv) if a.chain_map else {}

    mk = set(a.model_chains.split(",")) if a.model_chains else None
    rk = set(a.ref_chains.split(",")) if a.ref_chains else None
    m = ca_table(a.model, cmap, a.resmin, a.resmax, mk)
    r = ca_table(a.ref, {}, a.resmin, a.resmax, rk)
    keys = sorted(set(m) & set(r))
    same = [k for k in keys if m[k][0] == r[k][0]]
    print(f"[cmp] model={a.model}\n[cmp] ref  ={a.ref}")
    print(f"[cmp] model CA={len(m)} ref CA={len(r)} paired-by-number={len(keys)} identical-residue-names={len(same)}")
    if len(same) < 3:
        print("[cmp] VERDICT=NO_OVERLAP (numbering or chain mismatch; nothing compared)")
        return 2
    P = np.stack([m[k][1] for k in same])
    Q = np.stack([r[k][1] for k in same])
    rmsd, per = kabsch_rmsd(P, Q)
    gdt = {t: float((per <= t).mean()) for t in (1, 2, 4, 8)}
    gdt_ts = sum(gdt.values()) / 4
    bm = float(np.mean([m[k][2] for k in same]))
    br = float(np.mean([r[k][2] for k in same]))
    print(f"[cmp] CA_RMSD={rmsd:.3f} A over {len(same)} residues   "
          f"frac<=1A={gdt[1]:.3f} <=2A={gdt[2]:.3f} <=4A={gdt[4]:.3f} <=8A={gdt[8]:.3f}  GDT_TS~{gdt_ts:.3f}")
    print(f"[cmp] mean CA B-factor/pLDDT over paired: model={bm:.2f} ref={br:.2f} "
          f"mean|dB|={float(np.mean([abs(m[k][2]-r[k][2]) for k in same])):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
