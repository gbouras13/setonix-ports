#!/usr/bin/env python3
"""cfs_tokens.py INPUT -- the largest residue count (every copy of every chain) of a colabfold_batch input: an .a3m (ColabFold's complex
header `#len1,len2<TAB>card1,card2`, else the first sequence), a FASTA (chains of one complex joined by ':'), a .csv (id,sequence), or a
directory of them. Standard library only; prints one integer (0 when nothing is recognised). Used by cfs-run.sh to choose the launch
serialisation by size.
"""
import csv
import os
import sys


def a3m_tokens(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        first = f.readline().strip().lstrip("\x00")
        if first.startswith("#") and "\t" in first:
            lens, cards = first[1:].split("\t", 1)
            try:
                return sum(int(l) * int(c) for l, c in zip(lens.split(","), cards.split(",")))
            except ValueError:
                return 0
        seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq:
                    break
                continue
            seq.append(line)
        return sum(1 for ch in "".join(seq) if ch.isupper() or ch == "-")


def fasta_tokens(path):
    best, cur = 0, []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in list(f) + [">"]:
            line = line.strip()
            if line.startswith(">"):
                if cur:
                    best = max(best, len("".join(cur).replace(":", "").replace("/", "")))
                cur = []
            elif line:
                cur.append(line)
    return best


def csv_tokens(path):
    best = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[1] and row[1].lower() != "sequence":
                best = max(best, len(row[1].replace(":", "")))
    return best


def tokens(path):
    if os.path.isdir(path):
        return max([tokens(os.path.join(path, n)) for n in os.listdir(path)] or [0])
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".a3m":
            return a3m_tokens(path)
        if ext in (".fasta", ".fa", ".faa", ".fas"):
            return fasta_tokens(path)
        if ext == ".csv":
            return csv_tokens(path)
    except OSError:
        return 0
    return 0


if __name__ == "__main__":
    print(tokens(sys.argv[1]) if len(sys.argv) > 1 else 0)
