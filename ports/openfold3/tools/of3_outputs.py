#!/usr/bin/env python3
"""of3_outputs.py compare REF OUT [OUT ...] [--identical] -- OpenFold3 prediction directories, sample by sample.

A sample is <query>/seed_<s>/<query>_seed_<s>_sample_<k>_*: its structure (_model.cif), its aggregated confidences
(_confidences_aggregated.json) and its full confidences (_confidences.npz; float32 pLDDT / PAE / PDE, the port's runner yamls write them).
For each sample of REF, against the same sample in each OUT:
  - byte identity of the three files. None of them carries a time: the CIF has no date, the npz's zip entries carry a fixed one;
    _structure_first.json does carry write times, so it is compared by the model digest it records, not by its bytes;
  - when the structures differ, the CA RMSD after superposition (Kabsch), over the CA atoms the two share;
  - the change in mean pLDDT, pTM, ipTM and the ranking score.
Exit 0; with --identical, 1 unless every sample of REF is in each OUT and byte-identical there, and each OUT holds no other sample.
"""
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np

PAT = re.compile(r"^(?P<q>.+)_seed_(?P<s>\d+)_sample_(?P<k>\d+)_model\.cif$")


def samples(root):
    out = {}
    for cif in glob.glob(os.path.join(root, "**", "*_model.cif"), recursive=True):
        m = PAT.match(os.path.basename(cif))
        if m:
            out[(m.group("q"), int(m.group("s")), int(m.group("k")))] = cif[: -len("_model.cif")]
    return out


def same_bytes(a, b):
    if not (os.path.isfile(a) and os.path.isfile(b)):
        return None
    if os.path.getsize(a) != os.path.getsize(b):
        return False
    with open(a, "rb") as fa, open(b, "rb") as fb:
        while True:
            x, y = fa.read(1 << 20), fb.read(1 << 20)
            if x != y:
                return False
            if not x:
                return True


def ca(path):
    hdr, rows, inside = [], [], False
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("_atom_site."):
                inside = True
                hdr.append(s.split(".", 1)[1].strip())
                continue
            if inside:
                if not s or s.startswith("#") or s.startswith("loop_") or s.startswith("_"):
                    if rows:
                        break
                    continue
                rows.append(s.split())
    ix = {k: i for i, k in enumerate(hdr)}
    pts = {}
    for r in rows:
        if len(r) < len(hdr) or r[ix["label_atom_id"]].strip('"') != "CA":
            continue
        pts[(r[ix["label_asym_id"]], r[ix["label_seq_id"]])] = (float(r[ix["Cartn_x"]]), float(r[ix["Cartn_y"]]), float(r[ix["Cartn_z"]]))
    return pts


def rmsd(a, b):
    keys = sorted(set(a) & set(b))
    if len(keys) < 3:
        return None
    P = np.array([a[k] for k in keys])
    Q = np.array([b[k] for k in keys])
    P, Q = P - P.mean(0), Q - Q.mean(0)
    V, _, Wt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(V @ Wt))
    R = V @ np.diag([1.0, 1.0, d]) @ Wt
    return float(np.sqrt(((P @ R - Q) ** 2).sum(1).mean()))


def agg(prefix):
    try:
        return json.load(open(prefix + "_confidences_aggregated.json"))
    except (OSError, ValueError):
        return {}


def final_digest(prefix):
    try:
        return json.load(open(prefix + "_structure_first.json")).get("final_sha256")
    except (OSError, ValueError):
        return None


def compare(ref, out):
    R, O = samples(ref), samples(out)
    print("== %s against %s: %d sample(s), %d in OUT" % (out, ref, len(R), len(O)))
    n_same, rows = 0, []
    for key in sorted(R):
        q, s, k = key
        if key not in O:
            print("  %s seed %d sample %d: MISSING in OUT" % key)
            rows.append(False)
            continue
        a, b = R[key], O[key]
        cif = same_bytes(a + "_model.cif", b + "_model.cif")
        js = same_bytes(a + "_confidences_aggregated.json", b + "_confidences_aggregated.json")
        npz = same_bytes(a + "_confidences.npz", b + "_confidences.npz")
        ident = bool(cif) and js is not False and npz is not False
        for side, p in (("REF", a), ("OUT", b)):     # the model file must still be the one its run wrote (a truncated copy is not)
            rec = final_digest(p)
            if rec and rec != hashlib.sha256(open(p + "_model.cif", "rb").read()).hexdigest():
                print("  %s seed %d sample %d: %s's model file is not the one its run recorded (structure_first.json)" % (q, s, k, side))
                ident = False
        n_same += ident
        rows.append(ident)
        ca_r = None if cif else rmsd(ca(a + "_model.cif"), ca(b + "_model.cif"))
        ga, gb = agg(a), agg(b)
        dl = ["%s %+.4g" % (m, gb[m] - ga[m]) for m in ("avg_plddt", "ptm", "iptm", "sample_ranking_score")
              if isinstance(ga.get(m), (int, float)) and isinstance(gb.get(m), (int, float))]
        print("  %s seed %d sample %d: %s  model=%s aggregated=%s full=%s%s  %s"
              % (q, s, k, "IDENTICAL" if ident else "DIFFER", "same" if cif else "differs",
                 {True: "same", False: "differs", None: "absent"}[js], {True: "same", False: "differs", None: "absent"}[npz],
                 "" if ca_r is None else "  CA RMSD %.3f A" % ca_r, " ".join(dl) if not ident else ""))
    extra = sorted(set(O) - set(R))
    for key in extra:
        print("  %s seed %d sample %d: only in OUT" % key)
    all_same = bool(rows) and all(rows) and not extra
    print("  %d of %d sample(s) byte-identical (model, aggregated and full confidences)%s"
          % (n_same, len(R), "" if not extra else "; %d sample(s) only in OUT" % len(extra)))
    return all_same


if __name__ == "__main__":
    args = sys.argv[1:]
    strict = "--identical" in args
    args = [a for a in args if a != "--identical"]
    if len(args) < 3 or args[0] != "compare":
        sys.exit(__doc__)
    ok = all([compare(args[1], o) for o in args[2:]])
    sys.exit(0 if (ok or not strict) else 1)
