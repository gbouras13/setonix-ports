#!/usr/bin/env python
"""pms_designdiff.py -- two ProteinMPNN output folders, counted design by design.

    python pms_designdiff.py REF_DIR RUN_DIR [--label NAME]

For every `seqs/<name>.fa` of REF_DIR: each design record (`>T=..., sample=N, ...`) against RUN_DIR's record of the same item and
sample number. Prints one line:

    DESIGNDIFF <label> items=<n>/<n> designs=<identical>/<total> (<pct>%) flipped=<n designs> first_flip=<item>:<sample>@<position>
               score_max|d|=<x> recovery_ref=<x> recovery_run=<x>

A "flip" is a design whose sequence differs from the reference's: with routes that only reorder summations, a differing design means a
sampling draw landed on another residue (and everything after it in that design's decode order follows from there). `position` is the
first differing residue index (1-based, slashes between chains ignored). Exit 0 when every design matches, 1 otherwise, 2 on a
missing item. numpy only.
"""
import argparse
import os
import re
import sys

import numpy as np

HDR = re.compile(r"^>T=(?P<T>[^,]+), sample=(?P<sample>\d+), score=(?P<score>[^,]+), global_score=(?P<g>[^,]+), seq_recovery=(?P<rec>\S+)")


def designs(path):
    """{(T, sample): (seq, score, recovery)} of one .fa file."""
    out = {}
    lines = open(path).read().splitlines()
    for i in range(0, len(lines) - 1, 2):
        m = HDR.match(lines[i])
        if m:
            out[(m.group("T"), int(m.group("sample")))] = (lines[i + 1], float(m.group("score")), float(m.group("rec")))
    return out


def first_diff(a, b):
    a, b = a.replace("/", ""), b.replace("/", "")
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i + 1
    return len(a) + 1 if a != b else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("run"); ap.add_argument("--label", default="")
    a = ap.parse_args()
    items = sorted(f[:-3] for f in os.listdir(os.path.join(a.ref, "seqs")) if f.endswith(".fa"))
    n_items_ok = n_total = n_same = 0
    flips = []
    smax = 0.0
    recs = ([], [])
    for it in items:
        pr, pq = os.path.join(a.ref, "seqs", it + ".fa"), os.path.join(a.run, "seqs", it + ".fa")
        if not os.path.exists(pq):
            print("DESIGNDIFF %s MISSING %s in %s" % (a.label, it, a.run), flush=True)
            return 2
        n_items_ok += 1
        R, Q = designs(pr), designs(pq)
        for key, (seq, score, rec) in R.items():
            n_total += 1
            recs[0].append(rec)
            if key not in Q:
                flips.append((it, key[1], 0)); continue
            qseq, qscore, qrec = Q[key]
            recs[1].append(qrec)
            smax = max(smax, abs(score - qscore))
            if qseq == seq:
                n_same += 1
            else:
                flips.append((it, key[1], first_diff(seq, qseq)))
    f0 = "%s:%d@%d" % flips[0] if flips else "none"
    print("DESIGNDIFF %s items=%d/%d designs=%d/%d (%.4f%%) flipped=%d first_flip=%s score_max|d|=%.3e recovery_ref=%.4f recovery_run=%.4f" % (
        a.label or os.path.basename(a.run.rstrip("/")), n_items_ok, len(items), n_same, n_total, 100.0 * n_same / max(n_total, 1),
        len(flips), f0, smax, float(np.mean(recs[0])) if recs[0] else float("nan"), float(np.mean(recs[1])) if recs[1] else float("nan")), flush=True)
    return 0 if n_same == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
