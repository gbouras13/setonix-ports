#!/usr/bin/env python
"""Compare ProteinMPNN design outputs (upstream's layout: seqs/<name>.fa, scores/<name>.npz, probs/<name>.npz).

    python pms_compare.py A_DIR B_DIR [--json OUT.json] [--quiet]     A vs B, item by item
    python pms_compare.py --stats A_DIR                                 one folder's recovery / score statistics

Per item (the .fa stem):
  fa        'identical' (bytes) | 'differ'; when they differ: header lines equal?, sequences identical (n/m), mean per-site identity of
            the sampled sequences A vs B, scores |dA-dB| max
  scores    np.array_equal per array (score, global_score); max |d|
  probs     np.array_equal per array (probs, log_probs, S, mask; chain_order compared as lists); max |d| for float arrays
Summary line: n items, fa identical k/n, scores equal k/n, probs equal k/n, and (both folders) mean seq_recovery / score from the headers.
Exit 0 when every item present in A is byte-identical in B (fa) and every npz array is equal; 1 otherwise; 2 on usage errors.
Standard library + numpy only (runs on a CPU node, never on a login node).
"""
import argparse
import json
import os
import re
import sys

import numpy as np

HDR_RE = re.compile(r"^>T=(?P<T>[^,]+), sample=(?P<sample>\d+), score=(?P<score>[^,]+), global_score=(?P<gscore>[^,]+), seq_recovery=(?P<rec>\S+)")
NAT_RE = re.compile(r"^>(?P<name>[^,]+), score=(?P<score>[^,]+), global_score=(?P<gscore>[^,]+),")


def read_fa(path):
    """-> (native (header, seq) | None, [(header, seq, fields)] for the designs)."""
    with open(path) as fh:
        lines = [l.rstrip("\n") for l in fh]
    recs = [(lines[i], lines[i + 1] if i + 1 < len(lines) else "") for i in range(0, len(lines), 2)]
    native, designs = None, []
    for h, s in recs:
        m = HDR_RE.match(h)
        if m:
            designs.append((h, s, m.groupdict()))
        elif native is None and NAT_RE.match(h):
            native = (h, s)
    return native, designs


def site_identity(a, b):
    a, b = a.replace("/", ""), b.replace("/", "")
    if not a or len(a) != len(b):
        return float("nan")
    return sum(x == y for x, y in zip(a, b)) / len(a)


def cmp_npz(pa, pb):
    out = {"present": os.path.exists(pa) and os.path.exists(pb)}
    if not out["present"]:
        out["equal"] = os.path.exists(pa) == os.path.exists(pb)
        return out
    A, B = np.load(pa, allow_pickle=True), np.load(pb, allow_pickle=True)
    keys = sorted(set(A.files) | set(B.files))
    eq, md = {}, {}
    for k in keys:
        if k not in A.files or k not in B.files:
            eq[k] = False
            continue
        a, b = A[k], B[k]
        if a.dtype == object or b.dtype == object or a.dtype.kind in "US" or b.dtype.kind in "US":
            eq[k] = bool(a.shape == b.shape and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a.ravel(), b.ravel())))
            continue
        eq[k] = bool(a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b))
        if a.shape == b.shape and a.dtype.kind == "f":
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))
            md[k] = float(np.nanmax(d)) if d.size else 0.0
    out.update({"equal": all(eq.values()), "arrays": eq, "max_abs": md})
    return out


def stats(folder):
    recs, scores, nat = [], [], []
    if not os.path.isdir(os.path.join(folder, "seqs")):
        return {"n_designs": 0, "mean_recovery": None, "sd_recovery": None, "mean_score": None, "mean_native_score": None}
    for f in sorted(os.listdir(os.path.join(folder, "seqs"))):
        if not f.endswith(".fa"):
            continue
        native, designs = read_fa(os.path.join(folder, "seqs", f))
        for _, _, g in designs:
            recs.append(float(g["rec"])); scores.append(float(g["score"]))
        if native:
            nat.append(float(NAT_RE.match(native[0]).group("score")))
    return {"n_designs": len(recs), "mean_recovery": float(np.mean(recs)) if recs else None, "sd_recovery": float(np.std(recs)) if recs else None,
            "mean_score": float(np.mean(scores)) if scores else None, "mean_native_score": float(np.mean(nat)) if nat else None}


def compare(a_dir, b_dir):
    items = sorted(f[:-3] for f in os.listdir(os.path.join(a_dir, "seqs")) if f.endswith(".fa"))
    res = {}
    for it in items:
        pa, pb = os.path.join(a_dir, "seqs", it + ".fa"), os.path.join(b_dir, "seqs", it + ".fa")
        r = {}
        if not os.path.exists(pb):
            r["fa"] = "missing in B"
        else:
            ba, bb = open(pa, "rb").read(), open(pb, "rb").read()
            if ba == bb:
                r["fa"] = "identical"
            else:
                r["fa"] = "differ"
                (na, da), (nb, db) = read_fa(pa), read_fa(pb)
                r["native_header_equal"] = (na or ("",))[0] == (nb or ("",))[0]
                r["n_designs"] = (len(da), len(db))
                same = [x[1] == y[1] for x, y in zip(da, db)]
                r["seqs_identical"] = "%d/%d" % (sum(same), len(same))
                ids = [site_identity(x[1], y[1]) for x, y in zip(da, db)]
                r["mean_site_identity"] = float(np.nanmean(ids)) if ids else None
                ds = [abs(float(x[2]["score"]) - float(y[2]["score"])) for x, y in zip(da, db)]
                r["score_max_abs"] = max(ds) if ds else None
        for kind in ("scores", "probs"):
            c = cmp_npz(os.path.join(a_dir, kind, it + ".npz"), os.path.join(b_dir, kind, it + ".npz"))
            if c.get("present") or not c.get("equal", True):
                r[kind] = c
        res[it] = r
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b", nargs="?")
    ap.add_argument("--stats", action="store_true"); ap.add_argument("--json"); ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if a.stats or a.b is None:
        s = stats(a.a)
        print("STATS %s n_designs=%d mean_recovery=%.4f sd=%.4f mean_score=%.4f mean_native_score=%s" % (
            a.a, s["n_designs"], s["mean_recovery"] or float("nan"), s["sd_recovery"] or float("nan"), s["mean_score"] or float("nan"), s["mean_native_score"]))
        return 0
    res = compare(a.a, a.b)
    n = len(res)
    fa_id = sum(1 for r in res.values() if r.get("fa") == "identical")
    sc = [r["scores"]["equal"] for r in res.values() if "scores" in r]
    pr = [r["probs"]["equal"] for r in res.values() if "probs" in r]
    if not a.quiet:
        for it, r in res.items():
            if r.get("fa") != "identical" or not all(r[k]["equal"] for k in ("scores", "probs") if k in r):
                print("ITEM %s %s" % (it, json.dumps(r, default=str)))
    sa, sb = stats(a.a), stats(a.b)
    ok = fa_id == n and all(sc) and all(pr)
    print("COMPARE %s vs %s: items=%d fa_identical=%d/%d scores_equal=%d/%d probs_equal=%d/%d recovery A=%.4f B=%.4f score A=%.4f B=%.4f -> %s" % (
        a.a, a.b, n, fa_id, n, sum(sc), len(sc), sum(pr), len(pr), sa["mean_recovery"] or float("nan"), sb["mean_recovery"] or float("nan"),
        sa["mean_score"] or float("nan"), sb["mean_score"] or float("nan"), "BYTE-IDENTICAL" if ok else "DIFFER"))
    if a.json:
        json.dump({"items": res, "stats_a": sa, "stats_b": sb, "identical": ok}, open(a.json, "w"), indent=1, default=str)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
