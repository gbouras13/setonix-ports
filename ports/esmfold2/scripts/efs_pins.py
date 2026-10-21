#!/usr/bin/env python3
"""The kit's pin check with ONE change, for scripts/ef2-run.sh: the image-layer half REPORTS instead of refusing.

    python -I scripts/efs_pins.py <kit esmfold2 dir>        exit 0 = both upstream forks at their pinned commits; 3 = not

It loads the kit's own stock/check_pins.py (unmodified) and calls its two functions:
  check(PINS["upstream"])    the esm and transformers forks at their pinned commits -- REFUSED exactly as run.sh refuses (exit 3)
  image_pins(PINS["image"])  flash_attn / xformers / transformer_engine at their CUDA versions -- these are CUDA builds that do not
                             exist for ROCm, so on this machine they are absent by construction; the line is printed, the run goes on.
Upstream itself runs without them (its FLASH_ATTN_AVAILABLE / _te_available switches are False: SDPA atom attention, the pure-PyTorch
LayerNorm+MLP and the torch rotary in the LM), and the kit's documented setting for such a stack is ESMFOLD2_OPT_REQUIRE_FAST_ENV=0,
under which every route prints those words on its SETTINGS / ACTIVE line. Standard library only (run under python -I).
"""
import importlib.util
import json
import os
import sys


def main(argv):
    if len(argv) != 1:
        print("usage: python -I efs_pins.py <kit esmfold2 dir>", file=sys.stderr)
        return 2
    kit = argv[0]
    path = os.path.join(kit, "stock", "check_pins.py")
    spec = importlib.util.spec_from_file_location("esmfold2_check_pins", path)
    cp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cp)
    pins = json.load(open(os.path.join(kit, "stock", "PINS.json")))
    bad, _ = cp.check(pins["upstream"])
    ibad, idet = cp.image_pins(pins.get("image"))
    if ibad:
        got = ", ".join(f"{k}={d['got'] or 'absent'}" for k, d in idet.items())
        print(f"[efs] pins: image layer (stock/PINS.json \"image\": the CUDA accelerated layer) not installed on this ROCm stack ({got}) "
              f"-- reported, not refused (ef2-run.sh; ESMFOLD2_OPT_REQUIRE_FAST_ENV={os.environ.get('ESMFOLD2_OPT_REQUIRE_FAST_ENV', 'unset')})",
              file=sys.stderr)
    if bad:
        for b in bad:
            print(f"check_pins: {b}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
