#!/usr/bin/env python3
"""CA RMSD between two OpenDDE predictions of the SAME sample index.

The only valid structural comparison across runs is same-sample-index vs
same-sample-index: different diffusion samples within one run differ by tens of
angstroms, so comparing sample i to sample j measures nothing (af3-setonix
RUNTIME.md). Used here to check that row-sharding reproduces the single-GCD
answer, which is the sharding correctness claim.

  python tools/compare_structures.py a.cif b.cif
"""
import sys

import numpy as np
from biotite.structure.io.pdbx import CIFFile, get_structure


def ca(path):
    st = get_structure(CIFFile.read(path), model=1)
    sel = st[(st.atom_name == "CA")]
    return sel.coord, sel


def main():
    if len(sys.argv) != 3:
        return 2
    a, _ = ca(sys.argv[1])
    b, _ = ca(sys.argv[2])
    if a.shape != b.shape:
        print(f"SHAPE MISMATCH {a.shape} vs {b.shape}")
        return 1
    # superpose (Kabsch) before measuring: the model is not frame-pinned
    ac, bc = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(ac.T @ bc)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    rms_sup = float(np.sqrt(((ac @ r - bc) ** 2).sum(1).mean()))
    rms_raw = float(np.sqrt(((a - b) ** 2).sum(1).mean()))
    print(f"n_CA={len(a)}  CA_RMSD_superposed={rms_sup:.3f} A  CA_RMSD_raw={rms_raw:.3f} A")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
