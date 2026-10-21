#!/usr/bin/env python3
"""of3_query.py -- two small things the entry point needs from an OpenFold3 query JSON.

    of3_query.py QUERY.json                  prints "<residues> <msa>": residues as the launcher counts them (chain_ids x sequence
                                             length, over every chain with a sequence), and "all" when every protein chain carries
                                             main_msa_file_paths, else the number of protein chains without one
    of3_query.py --absolutize QUERY.json BASE   rewrites every relative path in the chains' fields that exists below BASE as an absolute
                                             path, in place: `run_openfold align-msa-server` writes them relative to its working
                                             directory, and the prediction ranks run elsewhere (the native port's prep did the same)
"""
import json
import os
import sys


def chains(query_set):
    for query in (query_set.get("queries") or {}).values():
        for chain in query.get("chains") or []:
            yield chain


def ids(chain):
    c = chain.get("chain_ids")
    if c is None:
        c = chain.get("chain_id")
    return [c] if isinstance(c, str) else list(c or [])


def summary(path):
    q = json.load(open(path))
    residues, missing = 0, 0
    for c in chains(q):
        residues += len(ids(c)) * len(c.get("sequence") or "")
        if (c.get("molecule_type") or "").lower() == "protein" and not c.get("main_msa_file_paths"):
            missing += 1
    print(residues, "all" if missing == 0 else missing)


def absolutize(path, base):
    q = json.load(open(path))
    n = 0

    def fix(v):
        nonlocal n
        if isinstance(v, str) and v and not os.path.isabs(v) and os.path.exists(os.path.join(base, v)):
            n += 1
            return os.path.normpath(os.path.join(base, v))
        if isinstance(v, list):
            return [fix(x) for x in v]
        return v

    for c in chains(q):
        for k in list(c.keys()):
            c[k] = fix(c[k])
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(q, fh, indent=1)
    os.replace(tmp, path)
    print("[openfold3] %s: %d path(s) made absolute below %s" % (path, n, base), file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        summary(sys.argv[1])
    elif len(sys.argv) == 4 and sys.argv[1] == "--absolutize":
        absolutize(sys.argv[2], os.path.abspath(sys.argv[3]))
    else:
        sys.exit(__doc__)
