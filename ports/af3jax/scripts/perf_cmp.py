#!/usr/bin/env python3
"""perf_cmp.py OUT REF NAME... [--core LO-HI] [--xtal PDB.cif[:CHAIN]] [--conf] -- one table for a same-input, same-seeds comparison of runs under OUT/<name>/ (logs OUT/<name>.log).

Per run: seed times from the runner's log (the first seed compiles, the second is warm), the allocator peak (``[peak]`` line), the warm
speed-up against REF, and the structures against REF's, three ways:
  core     each chain's own fold: per chain, CA of residues LO..HI superposed alone, against REF's SAME sample (same seed and index, so
           the same noise key); mean and max over chains x samples. Blind to where the chains sit, so it is the fold check. --core only.
  same     the whole complex against REF's same sample. Sensitive to the chains' arrangement: on a multi-modal target (the HK97 4-mer)
           a tiny numerical difference can send a sample to another arrangement, and this number jumps by tens of angstroms.
  nearest  each sample against the CLOSEST of REF's samples (any seed, any index): does the run produce arrangements REF produces?
  xtal     (--xtal PDB.cif[:CHAIN], with --core) every chain of every sample against the experimental chain, core residues only:
           the absolute accuracy check -- a lever that costs accuracy shows here, whatever the sampling noise does to the columns above.
plus the ranking scores (mean over samples, and the top one) and the mean CA pLDDT. The yardsticks come from REF itself: its sample-to-sample
spread (sample 0 vs the others, per seed) and, for 'nearest', each REF sample against its closest OTHER REF sample. A second stock run listed
among NAME gives the run-to-run noise of every column. --conf appends a confidence table from each sample's summary_confidences.json: mean
pTM and ipTM with the mean |difference| from REF's same sample, the samples flagged has_clash, and the samples with a non-finite pLDDT,
ranking score or pTM (opt-in, so callers that tail the output keep their lines). CPU only (gemmi, numpy).
"""
import csv
import glob
import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_structs import ca_table, kabsch_rmsd  # noqa: E402

_CA = {}


def cas(path):
    if path not in _CA:
        _CA[path] = ca_table(path, {}, None, None) if os.path.isfile(path) else {}
    return _CA[path]


def scores_file(d):
    """upstream names it ranking_scores.csv; this fork prefixes the job name (<job>_ranking_scores.csv)"""
    hits = sorted(glob.glob(os.path.join(d, "*ranking_scores.csv")))
    return hits[0] if hits else None


def job_dir(out, name):
    hits = sorted(glob.glob(os.path.join(out, name, "*", "*ranking_scores.csv")))
    return os.path.dirname(hits[0]) if hits else None


def sample_cif(d, s, k):
    j = os.path.basename(d)
    return os.path.join(d, f"seed-{s}_sample-{k}", f"{j}_seed-{s}_sample-{k}_model.cif")


def rmsd(a, b, sel=lambda key: True):
    m, r = cas(a), cas(b)
    keys = [k for k in sorted(set(m) & set(r)) if m[k][0] == r[k][0] and sel(k)]
    if len(keys) < 3:
        return float("nan")
    return kabsch_rmsd(np.stack([m[k][1] for k in keys]), np.stack([r[k][1] for k in keys]))[0]


def core_rmsds(a, b, lo, hi):
    """one RMSD per chain: that chain's residues lo..hi, superposed alone"""
    chains = sorted({c for c, _ in cas(b)})
    return [rmsd(a, b, lambda k, c=c: k[0] == c and lo <= k[1] <= hi) for c in chains]


def xtal_rmsds(a, xtal, xchain, lo, hi):
    """every chain of model a against the experimental chain xchain, core residues lo..hi, each superposed alone"""
    m, x = cas(a), cas(xtal)
    out = []
    for c in sorted({c for c, _ in m}):
        keys = [k for k in sorted(m) if k[0] == c and lo <= k[1] <= hi and (xchain, k[1]) in x and x[(xchain, k[1])][0] == m[k][0]]
        if len(keys) >= 3:
            out.append(kabsch_rmsd(np.stack([m[k][1] for k in keys]), np.stack([x[(xchain, k[1])][1] for k in keys]))[0])
    return out


def summary(d, s, k):
    j = os.path.basename(d)
    try:
        with open(os.path.join(d, f"seed-{s}_sample-{k}", f"{j}_seed-{s}_sample-{k}_summary_confidences.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def conf_table(out, ref, names, keys):
    num = lambda v: float(v) if isinstance(v, (int, float)) else float("nan")  # noqa: E731
    rd = job_dir(out, ref)
    rsum = {key: summary(rd, *key) for key in keys}
    print(f"confidence per sample (summary_confidences.json), |d| = mean |difference| from {ref}'s same sample:")
    print(f"{'run':<5} {'pTM mean':>9} {'|d pTM|':>8} {'ipTM mean':>10} {'|d ipTM|':>9} {'has_clash':>10} {'non-finite':>11} {'n':>4}")
    for n in [ref] + names:
        d = job_dir(out, n)
        if d is None:
            print(f"{n:<5} NO OUTPUT")
            continue
        sm = {key: summary(d, *key) for key in keys}
        ptm = [num(sm[key].get("ptm")) for key in keys]
        iptm = [num(sm[key].get("iptm")) for key in keys]
        dptm = [abs(num(sm[key].get("ptm")) - num(rsum[key].get("ptm"))) for key in keys]
        diptm = [abs(num(sm[key].get("iptm")) - num(rsum[key].get("iptm"))) for key in keys]
        clash = sum(1 for key in keys if num(sm[key].get("has_clash")) > 0)
        bad = sum(1 for key in keys if not all(math.isfinite(x) for x in
                  (num(sm[key].get("ptm")), num(sm[key].get("ranking_score")), plddt(sample_cif(d, *key)))))
        f3 = lambda v: f"{np.nanmean(v):.4f}" if not np.all(np.isnan(v)) else "-"  # noqa: E731
        dd = (lambda v: "(ref)") if n == ref else f3
        print(f"{n:<5} {f3(ptm):>9} {dd(dptm):>8} {f3(iptm):>10} {dd(diptm):>9} {clash:>10} {bad:>11} {sum(1 for key in keys if sm[key]):>4}")


def plddt(path):
    t = cas(path)
    return float(np.mean([v[2] for v in t.values()])) if t else float("nan")


def scores(d):
    with open(scores_file(d)) as f:
        return {(int(r["seed"]), int(r["sample"])): float(r["ranking_score"]) for r in csv.DictReader(f)}


def log_facts(out, name):
    try:
        txt = open(os.path.join(out, f"{name}.log"), errors="replace").read()
    except OSError:
        return {}, "", "", []
    t = {int(s): float(v) for s, v in re.findall(r"Running model inference with seed (\d+) took ([0-9.]+)", txt)}
    pk = re.findall(r"\[peak\] device=\d+ .*?peak_GiB=(\S+)", txt)
    inst = re.findall(r"\[perf-launch\] installed (.*)", txt)
    return t, (pk[-1] if pk else ""), (inst[-1] if inst else ""), re.findall(r"\[perf-launch\] (report .*)", txt)


def mm(v):
    v = np.asarray(v, float)
    return f"{np.nanmean(v):6.2f} / {np.nanmax(v):5.2f}" if v.size and not np.all(np.isnan(v)) else f"{'-':>14}"


def main():
    argv = sys.argv[1:]
    core = None
    if "--core" in argv:
        i = argv.index("--core")
        lo, hi = (int(x) for x in argv[i + 1].split("-"))
        core = (lo, hi)
        del argv[i:i + 2]
    conf = "--conf" in argv
    if conf:
        argv.remove("--conf")
    xtal = None
    if "--xtal" in argv:
        i = argv.index("--xtal")
        path, _, xc = argv[i + 1].partition(":")
        xtal = (path, xc or "A")
        del argv[i:i + 2]
        if not core:
            sys.exit("--xtal needs --core LO-HI")
    out, ref, names = argv[0], argv[1], argv[2:]
    rd = job_dir(out, ref)
    if rd is None:
        sys.exit(f"reference {ref}: no output -- nothing to compare against")
    rsc = scores(rd)
    seeds, samples = sorted({s for s, _ in rsc}), sorted({k for _, k in rsc})
    keys = [(s, k) for s in seeds for k in samples]
    rt = log_facts(out, ref)[0]
    warm_seed = max(seeds)
    ref_cifs = {key: sample_cif(rd, *key) for key in keys}
    print(f"reference {ref}: seeds {seeds}, samples {samples}; warm = seed {warm_seed}; core = "
          f"{'residues %d-%d of each chain, superposed alone' % core if core else 'off (--core LO-HI)'}")
    print(f"{'run':<5} {'seed1_s':>8} {'warm_s':>8} {'speedup':>8} {'peak':>6}  {'core mean/max':>14}  {'same mean/max':>14}  "
          f"{'nearest mean/max':>16}  {'rank mean (top)':>16} {'pLDDT':>6}" + (f"  {'xtal core mean (best)':>21}" if xtal else ""))
    for n in [ref] + names:
        t, pk, _, _ = log_facts(out, n)
        d = job_dir(out, n)
        cold, warm = t.get(min(seeds)), t.get(warm_seed)
        spd = f"{rt[warm_seed] / warm:.3f}x" if (warm and rt.get(warm_seed)) else "-"
        f = lambda v: f"{v:.1f}" if v else "-"  # noqa: E731
        if d is None:
            print(f"{n:<5} {f(cold):>8} {f(warm):>8} {spd:>8} {pk or '-':>6}  NO OUTPUT (see its log)")
            continue
        sc = {key: v for key, v in scores(d).items() if key in keys}   # the reference's seeds only (a run may have run more)
        cifs = {key: sample_cif(d, *key) for key in keys}
        rank = f"{np.mean(list(sc.values())):.4f} ({max(sc.values()):.4f})" if sc else "-"
        pl = np.nanmean([plddt(c) for c in cifs.values()])
        xt = ""
        if xtal:
            xs = [v for c in cifs.values() for v in xtal_rmsds(c, xtal[0], xtal[1], *core)]
            xt = f"  {np.mean(xs):8.3f} ({np.min(xs):.3f}) n={len(xs)}" if xs else "  -"
        if n == ref:
            near = [min(rmsd(cifs[a], ref_cifs[b]) for b in keys if b != a) for a in keys]
            print(f"{n:<5} {f(cold):>8} {f(warm):>8} {'1.000x':>8} {pk or '-':>6}  {'(reference)':>14}  {'(reference)':>14}  "
                  f"{mm(near):>16}  {rank:>16} {pl:6.2f}{xt}   <- nearest: vs the closest OTHER {ref} sample")
            continue
        cr = [x for key in keys for x in core_rmsds(cifs[key], ref_cifs[key], *core)] if core else []
        same = [rmsd(cifs[key], ref_cifs[key]) for key in keys]
        near = [min(rmsd(cifs[a], ref_cifs[b]) for b in keys) for a in keys]
        print(f"{n:<5} {f(cold):>8} {f(warm):>8} {spd:>8} {pk or '-':>6}  {mm(cr):>14}  {mm(same):>14}  {mm(near):>16}  {rank:>16} {pl:6.2f}{xt}")
    sp_same = [rmsd(ref_cifs[(s, 0)], ref_cifs[(s, k)]) for s in seeds for k in samples if k]
    msg = f"yardsticks from {ref} itself (sample 0 vs the others, per seed): complex {mm(sp_same).strip()} A"
    if core:
        sp_core = [x for s in seeds for k in samples if k for x in core_rmsds(ref_cifs[(s, k)], ref_cifs[(s, 0)], *core)]
        msg += f"; core {mm(sp_core).strip()} A"
    print(msg)
    for n in [ref] + names:
        _, _, inst, reps = log_facts(out, n)
        if inst or reps:
            print(f"  {n:<5} installed: {inst or '-'}")
            for r in reps:
                print(f"  {n:<5}   {r[:220]}")
    if conf:
        conf_table(out, ref, names, keys)


if __name__ == "__main__":
    main()
