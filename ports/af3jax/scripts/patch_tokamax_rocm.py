#!/usr/bin/env python3
"""patch_tokamax_rocm.py -- make tokamax 0.0.12 answer honestly on ROCm (AMD) devices.

On ROCm, jax reports `compute_capability` as the GCN arch NAME -- 'gfx90a' on an MI250X
(measured: job 49861629). tokamax assumes it is an NVIDIA number and does float() on it in
four places on AlphaFold 3's path. Each one was hit in turn by a real prediction on Setonix:

  gpu_utils._compute_capability   float(getattr(device, 'compute_capability', None))
  gpu_utils.has_triton_support    float(device.compute_capability) >= 8.0    <- job 49861905 step A2
  gpu_utils.has_mosaic_gpu_support float(device.compute_capability) >= 9.0
  precision.to_dot_algorithm_preset  float(compute_capability) < 8.0        <- job 49861905 step B

(The fifth read, ops/ragged_dot/pallas_mosaic_gpu_common.py, is Mosaic-GPU-only MoE code that
AlphaFold 3 never calls and that jax refuses on ROCm anyway; left alone.)

For a device whose capability is a 'gfx...' string the patch makes them say:
  _compute_capability      -> None   (no NVIDIA capability: is_sm80/90/100 are False)
  has_mosaic_gpu_support   -> False  (jax's mosaic/gpu/core.py refuses ROCm by name)
  has_triton_support       -> os.environ TOKAMAX_ROCM_TRITON == "1"  (default False: measure first)
  to_dot_algorithm_preset  -> a new backend 'rocm': bf16 dots BF16_BF16_F32 (CDNA2 has bf16
                              matrix cores), f32 dots F32_F32_F32 at every precision

That last line is a judgement, stated here so it is reviewed rather than inherited. tokamax's
'gpu' backend maps f32 dots at DEFAULT/HIGH to TF32_TF32_F32 -- NVIDIA TensorFloat-32, which
gfx90a does not implement. Its 'gpu_old' backend (pre-Ampere) would avoid TF32 but also
upcasts every bf16 dot to f32, discarding the bf16 matrix cores. Neither fits CDNA2, so 'rocm'
takes 'gpu''s bf16 row and 'cpu''s f32 row. Net numerical effect vs an H100 reference: bf16
dots identical in kind; f32 dots MORE precise here (23-bit mantissa instead of TF32's 10).
NVIDIA devices take exactly the original code paths.

    python patch_tokamax_rocm.py <site-packages>            apply (both files)
    python patch_tokamax_rocm.py <site-packages> --check
    python patch_tokamax_rocm.py <site-packages> --revert
    python patch_tokamax_rocm.py <site-packages> --selftest  simulated gfx90a / '9.0' devices -- run it in a JOB
"""
import argparse
import importlib
import os
import shutil
import sys

MARK = "# [af3jax-setonix] ROCm-aware"

FILES = {
    os.path.join("tokamax", "_src", "gpu_utils.py"): [
        ("""  if device.platform != 'gpu':
    return None

  return float(getattr(device, 'compute_capability', None))""",
         """  if device.platform != 'gpu':
    return None

  cc = getattr(device, 'compute_capability', None)   """ + MARK + """
  if cc is None or str(cc).startswith('gfx'):        # ROCm: a GCN arch name, not an NVIDIA capability
    return None
  return float(cc)"""),
        ("""  if not hasattr(device, 'compute_capability'):
    return True

  # Only currently supported for Ampere and above.
  return float(device.compute_capability) >= 8.0""",
         """  if not hasattr(device, 'compute_capability'):
    return True

  # Only currently supported for Ampere and above.
  if str(device.compute_capability).startswith('gfx'):   """ + MARK + """
    import os as _os                                       # ROCm: opt-in, pending measurement on this arch
    return _os.environ.get('TOKAMAX_ROCM_TRITON', '0') == '1'
  return float(device.compute_capability) >= 8.0"""),
        ("""  # Only currently supported for Hopper and above.
  return float(device.compute_capability) >= 9.0""",
         """  # Only currently supported for Hopper and above.
  if str(getattr(device, 'compute_capability', '')).startswith('gfx'):   """ + MARK + """
    return False                                                          # jax: Mosaic GPU does not support ROCm
  return float(device.compute_capability) >= 9.0"""),
    ],
    os.path.join("tokamax", "_src", "precision.py"): [
        ("""    elif float(compute_capability) < 8.0:
      backend = "gpu_old\"""",
         """    elif str(compute_capability).startswith("gfx"):   """ + MARK + """
      backend = "rocm"   # CDNA: bf16 matrix cores yes (BF16_BF16_F32), TF32 no (f32 dots F32_F32_F32)
    elif float(compute_capability) < 8.0:
      backend = "gpu_old\""""),
        ("""    cpu={
        Precision.DEFAULT: "F32_F32_F32",
        Precision.HIGH: "F32_F32_F32",
        Precision.HIGHEST: "F32_F32_F32",
    },
)""",
         """    cpu={
        Precision.DEFAULT: "F32_F32_F32",
        Precision.HIGH: "F32_F32_F32",
        Precision.HIGHEST: "F32_F32_F32",
    },
    rocm={   """ + MARK + """: gfx90a has no TensorFloat-32, so no TF32 row here
        Precision.DEFAULT: "F32_F32_F32",
        Precision.HIGH: "F32_F32_F32",
        Precision.HIGHEST: "F32_F32_F32",
    },
)"""),
    ],
}


class FakeDevice:
    def __init__(self, cc):
        self.platform, self.compute_capability = "gpu", cc


def selftest(site):
    sys.path.insert(0, site)
    from absl import flags as _absl_flags                 # tokamax's config is absl flags
    if not _absl_flags.FLAGS.is_parsed():
        _absl_flags.FLAGS([sys.argv[0]])
    import jax                                            # noqa: PLC0415
    import jax.numpy as jnp                               # noqa: PLC0415
    import tokamax._src.gpu_utils as g                    # noqa: PLC0415
    import tokamax._src.precision as pr                   # noqa: PLC0415
    importlib.reload(g); importlib.reload(pr)
    state = {f: (MARK in open(os.path.join(site, f)).read()) for f in FILES}
    print("selftest:", ", ".join(f"{os.path.basename(f)}={'PATCHED' if v else 'UNPATCHED'}" for f, v in state.items()))
    amd, h100 = FakeDevice("gfx90a"), FakeDevice("9.0")

    def cell(fn):
        try:
            return repr(fn())
        except Exception as e:                            # noqa: BLE001 -- the failure IS the reading
            return f"RAISED {type(e).__name__}: {str(e)[:60]}"

    rows = [(name, lab, cell(lambda n=name, d=dev: getattr(g, n)(d)))
            for name in ("_compute_capability", "is_sm80", "is_sm90", "has_triton_support", "has_mosaic_gpu_support")
            for lab, dev in (("gfx90a", amd), ("9.0", h100))]
    # precision reads jax.default_backend() / jax.devices() itself: simulate both
    real_b, real_d = jax.default_backend, jax.devices
    try:
        for lab, dev in (("gfx90a", amd), ("9.0", h100)):
            jax.default_backend, jax.devices = (lambda: "gpu"), (lambda *a, **k: [dev])
            for dt in ("bfloat16", "float32"):
                for prec in (jax.lax.Precision.DEFAULT, jax.lax.Precision.HIGHEST):
                    rows.append((f"to_dot_algorithm_preset({dt},{prec.name})", lab,
                                 cell(lambda dt=dt, prec=prec: pr.to_dot_algorithm_preset(getattr(jnp, dt), getattr(jnp, dt), prec).name)))
    finally:
        jax.default_backend, jax.devices = real_b, real_d
    for r in rows:
        print(f"  {r[0]:42s} {r[1]:8s} {r[2]}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("site")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest(a.site)
    rc = 0
    for rel, edits in FILES.items():
        p = os.path.join(a.site, rel)
        if not os.path.isfile(p):
            sys.exit(f"patch_tokamax_rocm: no {rel} under {a.site}")
        src = open(p, encoding="utf-8").read()
        applied = MARK in src
        bak = p + ".rocm-orig"
        if a.check:
            print(f"{'APPLIED' if applied else 'NOT APPLIED'}: {p}")
            rc |= 0 if applied else 1
        elif a.revert:
            if not applied:
                print(f"NOT APPLIED, nothing to revert: {p}")
            elif not os.path.isfile(bak):
                sys.exit(f"patch_tokamax_rocm: no backup at {bak}")
            else:
                shutil.copyfile(bak, p)
                print(f"REVERTED from {bak}: {p}")
        elif applied:
            print(f"ALREADY APPLIED: {p}")
        else:
            for old, _ in edits:
                if old not in src:
                    sys.exit(f"patch_tokamax_rocm: anchor not found in {p} (tokamax moved?):\n{old}")
            shutil.copyfile(p, bak)
            for old, new in edits:
                src = src.replace(old, new, 1)
            open(p, "w", encoding="utf-8").write(src)
            print(f"APPLIED {len(edits)} edit(s) (backup at {bak}): {p}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
