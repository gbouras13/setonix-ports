#!/usr/bin/env python3
"""Make OpenFold3's hardcoded gfx90a Triton constants settable from the environment.

    python tools/make_tunable_gfx90a.py --apply   <path-to-openfold3-checkout>
    python tools/make_tunable_gfx90a.py --revert  <path-to-openfold3-checkout>

OpenFold3's forward triangle-attention kernel bakes its AMD tunings in as locals
(core/kernels/triton/evoformer.py), tuned on MI300X/gfx942:

    waves_per_eu = 3 if DIM <= 64 else 2
    block_size_q = 64
    _oss_bkv     = 16

On Setonix (MI250X/gfx90a) that combination reaches ~12% of the GCD's bf16 peak,
and the kernel is compute-bound rather than bandwidth-bound (~1.4% of HBM
bandwidth), so the ceiling is occupancy/tiling -- i.e. exactly these constants.

This patch does not change any default. It makes each constant readable from an
env var, falling back to the upstream value, so tools/sweep_gfx90a.py can search
without re-editing source between runs. It also exposes matrix_instr_nonkdim,
which upstream sets on the BACKWARD kernels but never on the forward -- and
inference only runs the forward.

The checkout is a git clone, so --revert is just `git checkout` of the one file.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REL = Path("openfold3/core/kernels/triton/evoformer.py")

OLD_WPE = """            extra_kern_args = {}
            if is_hip():
                waves_per_eu = 3 if DIM <= 64 else 2
                extra_kern_args = {
                    "waves_per_eu": waves_per_eu,
                    "allow_flush_denorm": True,
                }"""

NEW_WPE = '''            extra_kern_args = {}
            if is_hip():
                # af3-setonix: upstream default preserved when the env is unset.
                waves_per_eu = int(
                    os.environ.get("OF3_TUNE_WPE", "3" if DIM <= 64 else "2")
                )
                extra_kern_args = {
                    "waves_per_eu": waves_per_eu,
                    "allow_flush_denorm": True,
                }
                # Upstream sets matrix_instr_nonkdim on the BACKWARD kernels only;
                # the forward never selects an MFMA tile. Inference is forward-only,
                # so this is an untried axis on gfx90a. Unset => upstream behaviour.
                _mid = os.environ.get("OF3_TUNE_MFMA")
                if _mid:
                    extra_kern_args["matrix_instr_nonkdim"] = int(_mid)'''

OLD_TILES = """            # Forward tiles Q=64, KV=16: validated fastest on real (low_mem chunked)
            block_size_q = 64
            _oss_bkv = 16"""

NEW_TILES = '''            # Forward tiles Q=64, KV=16: validated fastest on real (low_mem chunked)
            # -- on MI300X/gfx942. af3-setonix makes them searchable on gfx90a;
            # unset env => upstream values exactly.
            block_size_q = int(os.environ.get("OF3_TUNE_BQ", "64"))
            _oss_bkv = int(os.environ.get("OF3_TUNE_BKV", "16"))'''


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkout", type=Path, help="the openfold3 git checkout")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--apply", action="store_true")
    g.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    target = args.checkout / REL
    if not target.is_file():
        print(f"not found: {target}", file=sys.stderr)
        return 1

    if args.revert:
        r = subprocess.run(
            ["git", "-C", str(args.checkout), "checkout", "--", str(REL)],
            capture_output=True, text=True,
        )
        if r.returncode:
            print(r.stderr, file=sys.stderr)
            return 1
        print(f"reverted {REL}")
        return 0

    src = target.read_text()
    if "OF3_TUNE_WPE" in src:
        print("already patched")
        return 0

    for old in (OLD_WPE, OLD_TILES):
        if old not in src:
            print("upstream text not found -- OpenFold3 version differs from the "
                  "one this patch was written against (v0.5.0 / c477165).",
                  file=sys.stderr)
            return 1

    src = src.replace(OLD_WPE, NEW_WPE).replace(OLD_TILES, NEW_TILES)
    if "\nimport os" not in src and "\nimport os\n" not in src:
        print("warning: `import os` not obviously present; kernel file uses "
              "os.environ elsewhere so this should be fine.", file=sys.stderr)
    target.write_text(src)
    print(f"patched {REL}")
    print("env knobs: OF3_TUNE_WPE, OF3_TUNE_BQ, OF3_TUNE_BKV, OF3_TUNE_MFMA")
    print("defaults unchanged when unset.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
