#!/usr/bin/env python3
"""cf_outputs.py -- ColabFold result directories: summaries, run-against-reference comparisons, crystal comparisons. CPU only (numpy +
Biopython from the port's venv).

  cf_outputs.py summary DIR [DIR ...]
      one line per ranked model: job, model/seed key, mean pLDDT, pTM, ipTM, ipSAE (when written), recycles (log.txt), finite?
  cf_outputs.py compare REF DIR
      per model key (model_<n>_seed_<sss>, independent of the rank): ATOM records byte-identical?, scores JSON identical?, CA RMSD after
      superposition (whole complex) and the worst chain's own CA RMSD, max |dpLDDT|, dpTM, dipTM, max |dPAE| (the JSON's 2-decimal values)
  cf_outputs.py xtal DIR REF.cif[:CHAIN] [--core LO-HI]
      every chain of every model against the experimental chain, residues LO..HI, each chain superposed alone (paired by residue number
      AND name; a numbering mismatch prints as 0 paired, never as a wrong RMSD)
"""
import glob
import json
import math
import os
import re
import sys

import numpy as np

KEY_RE = re.compile(r"_(?:unrelaxed|relaxed)_rank_(\d+)_(.+?)_(model_\d+_seed_\d+)\.pdb$")


def models(d):
    """{(job, key): {"pdb": path, "rank": n, "scores": path or None}} for every ranked PDB under d (non-recursive, then one level down)."""
    out = {}
    for pdb in sorted(glob.glob(os.path.join(d, "*_rank_*_model_*.pdb")) + glob.glob(os.path.join(d, "*", "*_rank_*_model_*.pdb"))):
        m = KEY_RE.search(os.path.basename(pdb))
        if not m:
            continue
        job = os.path.basename(pdb)[: m.start()]
        rank, mtype, key = int(m.group(1)), m.group(2), m.group(3)
        sc = os.path.join(os.path.dirname(pdb), f"{job}_scores_rank_{m.group(1)}_{mtype}_{key}.json")
        out[(job, key)] = {"pdb": pdb, "rank": rank, "scores": sc if os.path.isfile(sc) else None, "dir": os.path.dirname(pdb)}
    return out


def atoms(pdb):
    return [ln for ln in open(pdb, encoding="utf-8", errors="replace") if ln.startswith(("ATOM", "HETATM"))]


def ca_coords(pdb):
    """[(chain, resnum, resname, xyz, b)] of CA atoms from a ColabFold PDB."""
    out = []
    for ln in atoms(pdb):
        if ln[12:16].strip() != "CA":
            continue
        out.append((ln[21], int(ln[22:26]), ln[17:20].strip(), np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]), float(ln[60:66])))
    return out


def kabsch(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    diff = Pc @ R - Qc
    return float(np.sqrt((diff ** 2).sum(1).mean()))


def load_scores(path):
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def finite_pdb(pdb):
    for ln in atoms(pdb):
        try:
            vals = [float(ln[30:38]), float(ln[38:46]), float(ln[46:54]), float(ln[60:66])]
        except ValueError:
            return False
        if not all(math.isfinite(v) for v in vals):
            return False
    return True


def recycles(d, job):
    """the per-model recycle counts colabfold logged (log.txt: '<model> recycle=<n> ...' lines, the last per model)."""
    log = os.path.join(d, "log.txt")
    if not os.path.isfile(log):
        return {}
    last = {}
    for ln in open(log, encoding="utf-8", errors="replace"):
        m = re.search(r"(alphafold2\w*_model_\d+_seed_\d+) recycle=(\d+)", ln)
        if m:
            last[m.group(1)] = int(m.group(2))
    return last


def label(d):
    """the attempt's name: the directory's own name, or its parent's when it is colabfold's results directory 'out'."""
    d = os.path.normpath(d)
    return os.path.basename(os.path.dirname(d)) if os.path.basename(d) == "out" else os.path.basename(d)


def cmd_summary(dirs):
    for d in dirs:
        ms = models(d)
        if not ms:
            print(f"SUMMARY {label(d)}: no ranked model")
            continue
        for (job, key), m in sorted(ms.items(), key=lambda kv: (kv[0][0], kv[1]["rank"])):
            s = load_scores(m["scores"])
            pl = s.get("plddt") or []
            rc = recycles(m["dir"], job)
            rr = next((v for k, v in rc.items() if k.endswith(key)), None)
            ips = s.get("ipsae")
            ips = (max(v for row in ips for v in row if v is not None) if isinstance(ips, list) and ips and isinstance(ips[0], list) else ips)
            print(f"SUMMARY {label(d)} job={job} rank={m['rank']} {key} plddt={np.mean(pl) if pl else float('nan'):.2f} "
                  f"ptm={s.get('ptm')} iptm={s.get('iptm')} ipsae={ips} recycles={rr} finite={finite_pdb(m['pdb'])}")


def cmd_compare(ref, d):
    a, b = models(ref), models(d)
    keys = sorted(set(a) & set(b))
    if not keys:
        print(f"COMPARE {label(d)} vs {label(ref)}: no common model key (ref {sorted(a)[:3]}, run {sorted(b)[:3]})")
        return 1
    for k in keys:
        A, B = a[k], b[k]
        same_atoms = atoms(A["pdb"]) == atoms(B["pdb"])
        sa, sb = load_scores(A["scores"]), load_scores(B["scores"])
        same_scores = sa == sb
        ca, cb = ca_coords(A["pdb"]), ca_coords(B["pdb"])
        if len(ca) != len(cb):
            print(f"COMPARE {k}: CA count differs {len(ca)} vs {len(cb)}")
            continue
        P, Q = np.stack([x[3] for x in ca]), np.stack([x[3] for x in cb])
        whole = kabsch(P, Q)
        per_chain = {}
        for ch in sorted({x[0] for x in ca}):
            idx = [i for i, x in enumerate(ca) if x[0] == ch]
            per_chain[ch] = kabsch(P[idx], Q[idx])
        worst = max(per_chain.items(), key=lambda kv: kv[1])
        dpl = float(np.max(np.abs(np.array(sa.get("plddt", [0])) - np.array(sb.get("plddt", [0]))))) if sa.get("plddt") and sb.get("plddt") else float("nan")
        dpae = (float(np.max(np.abs(np.array(sa["pae"]) - np.array(sb["pae"])))) if sa.get("pae") and sb.get("pae") else float("nan"))
        f = lambda x, y: (None if x is None or y is None else round(float(y) - float(x), 3))  # noqa: E731
        print(f"COMPARE {label(d)} vs {label(ref)} {k[1]} rank {A['rank']}->{B['rank']} atoms_identical={same_atoms} scores_identical={same_scores} "
              f"ca_rmsd={whole:.3f} worst_chain={worst[0]}:{worst[1]:.3f} max_dplddt={dpl:.2f} dptm={f(sa.get('ptm'), sb.get('ptm'))} "
              f"diptm={f(sa.get('iptm'), sb.get('iptm'))} max_dpae={dpae:.2f}")
    return 0


def xtal_ca(path, chain):
    from Bio.PDB import MMCIFParser, PDBParser
    p = MMCIFParser(QUIET=True) if path.endswith(".cif") else PDBParser(QUIET=True)
    st = p.get_structure("x", path)
    out = {}
    for ch in st[0]:
        if ch.id != chain:
            continue
        for r in ch:
            if r.id[0] != " " or "CA" not in r:
                continue
            out[r.id[1]] = (r.get_resname(), r["CA"].coord.astype(float))
    return out


def cmd_xtal(d, ref, core):
    path, _, chain = ref.partition(":")
    chain = chain or "A"
    X = xtal_ca(path, chain)
    lo, hi = core if core else (min(X), max(X))
    for (job, key), m in sorted(models(d).items(), key=lambda kv: (kv[0][0], kv[1]["rank"])):
        ca = ca_coords(m["pdb"])
        res = []
        for ch in sorted({x[0] for x in ca}):
            pairs = [(x[3], X[x[1]][1]) for x in ca if x[0] == ch and lo <= x[1] <= hi and x[1] in X and X[x[1]][0] == x[2]]
            if len(pairs) < 3:
                res.append(f"{ch}:n=0")
                continue
            res.append(f"{ch}:{kabsch(np.stack([p for p, _ in pairs]), np.stack([q for _, q in pairs])):.2f}(n={len(pairs)})")
        print(f"XTAL {label(d)} job={job} rank={m['rank']} {key} core={lo}-{hi} vs {os.path.basename(path)}:{chain} " + " ".join(res))


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "summary":
        cmd_summary(rest)
        return 0
    if cmd == "compare":
        return cmd_compare(rest[0], rest[1])
    if cmd == "xtal":
        core = None
        if "--core" in rest:
            i = rest.index("--core")
            lo, hi = rest[i + 1].split("-")
            core = (int(lo), int(hi))
            rest = rest[:i] + rest[i + 2:]
        cmd_xtal(rest[0], rest[1], core)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
