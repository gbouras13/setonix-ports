#!/usr/bin/env python3
"""patch_rocm_capability.py -- teach stock AlphaFold 3 (JAX) that a ROCm device's
`compute_capability` is a GCN arch name, not an NVIDIA compute capability.

The defect, in stock `run_alphafold.py`:

    compute_capability = float(gpu_devices[_GPU_DEVICE.value].compute_capability)

On CUDA that attribute is "9.0" / "8.0". On ROCm, JAX's own code says it is the GCN arch
name -- `jax/experimental/mosaic/gpu/core.py` raises when it `.startswith("gfx")`, and
`jax/_src/pallas/triton/pallas_call_registration.py` sets `compute_capability = 0` on the
rocm lowering platform rather than parsing it. So `float()` raises ValueError on an
MI250X, at startup, before any GPU work.

What this patch does, and deliberately does NOT do: it makes the three checks that follow
(cc < 6.0; 7.0 <= cc < 8.0 needing an XLA flag and xla attention) apply only when the
attribute really is an NVIDIA capability, and it names the AMD device on stderr instead of
silently pretending. It does NOT invent a capability number for gfx90a -- that is exactly
the (9, 0) trap the PyTorch kits fall into, and inventing 9.0 here would hand an MI250X
every Hopper code path in the stack.

    python patch_rocm_capability.py <checkout>            apply
    python patch_rocm_capability.py <checkout> --check    is it applied?
    python patch_rocm_capability.py <checkout> --revert   put it back
"""
import argparse
import os
import shutil
import sys

TARGET = "run_alphafold.py"
MARK = "# [af3jax-setonix] ROCm capability guard"

OLD = """      if gpu_devices:
        compute_capability = float(
            gpu_devices[_GPU_DEVICE.value].compute_capability
        )
        if compute_capability < 6.0:"""

NEW = """      if gpu_devices:
        _raw_cc = gpu_devices[_GPU_DEVICE.value].compute_capability   """ + MARK + """
        if str(_raw_cc).startswith('gfx'):
          # AMD ROCm: `compute_capability` is the GCN arch name (jax's own
          # mosaic/gpu/core.py and pallas/triton/pallas_call_registration.py both
          # branch on exactly this), not an NVIDIA compute capability. The three
          # checks below are NVIDIA-specific, so they are skipped rather than
          # evaluated against a number this device does not have. No capability is
          # substituted: pretending an MI250X is compute capability 9.0 is how the
          # PyTorch stacks end up running Hopper kernel tables on CDNA2.
          print(
              f'AMD ROCm device detected (compute_capability={_raw_cc!r}); skipping'
              ' the NVIDIA compute-capability checks.'
          )
          compute_capability = None
        else:
          compute_capability = float(_raw_cc)
        if compute_capability is not None and compute_capability < 6.0:"""

OLD2 = """        elif 7.0 <= compute_capability < 8.0:"""
NEW2 = """        elif compute_capability is not None and 7.0 <= compute_capability < 8.0:"""


def path_of(checkout):
    p = os.path.join(checkout, TARGET)
    if not os.path.isfile(p):
        sys.exit(f"patch_rocm_capability: no {TARGET} under {checkout}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkout")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()

    p = path_of(a.checkout)
    src = open(p, encoding="utf-8").read()
    applied = MARK in src

    if a.check:
        print(f"{'APPLIED' if applied else 'NOT APPLIED'}: {p}")
        return 0 if applied else 1

    if a.revert:
        if not applied:
            print(f"NOT APPLIED, nothing to revert: {p}")
            return 0
        bak = p + ".rocm-orig"
        if not os.path.isfile(bak):
            sys.exit(f"patch_rocm_capability: no backup at {bak}; revert by hand")
        shutil.copyfile(bak, p)
        print(f"REVERTED from {bak}: {p}")
        return 0

    if applied:
        print(f"ALREADY APPLIED: {p}")
        return 0
    for needle in (OLD, OLD2):
        if needle not in src:
            sys.exit(f"patch_rocm_capability: anchor not found in {p}; upstream moved:\n{needle[:120]}")
    shutil.copyfile(p, p + ".rocm-orig")
    src = src.replace(OLD, NEW, 1).replace(OLD2, NEW2, 1)
    open(p, "w", encoding="utf-8").write(src)
    print(f"APPLIED (backup at {p}.rocm-orig): {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
