#!/usr/bin/env python3
"""prune_rocm_arch.py --keep gfx90a [--dry-run] DIR... -- delete GPU-kernel files built for OTHER architectures from ROCm library directories
(hipBLASLt alone installs 4.65 GiB, mostly kernels for other GPUs). Off unless the image is built with PRUNE_GPU_ARCH.

A regular file is removed only when its name carries at least one specific architecture token and none of them is the kept one. A name with
no token, or only generic targets (gfx9-generic, gfx11-generic, ...), stays; so does anything the pattern cannot parse -- every doubt keeps
the file. Token: 'gfx' + four digits, three digits, or two digits and a letter, optionally followed by a 2-3 hex-digit compute-unit count
(MIOpen names its databases gfx90a6e.kdb and the like). Note gfx90a is TWO digits plus 'a': a [0-9]{3,4} class alone misses it
(the method note in AF3JAX_PORT.md). Prints per directory what it removed and the bytes freed. Missing directories are skipped.
"""
import argparse
import os
import re
import sys

TOKEN = re.compile(r"gfx(\d{4}|\d{3}|\d{2}[a-z])(?:[0-9a-f]{2,3})?(?![0-9a-z])")


def arches(name):
    return {"gfx" + m.group(1) for m in TOKEN.finditer(name.lower())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", required=True, help="the architecture to keep, e.g. gfx90a")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("dirs", nargs="+")
    a = ap.parse_args()
    keep = a.keep.lower()
    if not re.fullmatch(r"gfx(\d{4}|\d{3}|\d{2}[a-z])", keep):
        sys.exit(f"--keep {a.keep}: not an architecture name")
    total = 0
    for d in a.dirs:
        if not os.path.isdir(d):
            print(f"[prune] skip {d}: not a directory")
            continue
        n = size = kept = 0
        for root, _, files in os.walk(d):
            for f in files:
                p = os.path.join(root, f)
                if os.path.islink(p) or not os.path.isfile(p):
                    continue
                tok = arches(f)
                if not tok or keep in tok or keep in f.lower():       # the literal name anywhere: always kept
                    kept += 1
                    continue
                size += os.path.getsize(p)
                n += 1
                if not a.dry_run:
                    os.remove(p)
        total += size
        print(f"[prune] {d}: {'would remove' if a.dry_run else 'removed'} {n} files ({size / 2**30:.2f} GiB), kept {kept}")
    print(f"[prune] total {'would free' if a.dry_run else 'freed'} {total / 2**30:.2f} GiB (kept architecture {keep})")


if __name__ == "__main__":
    main()
