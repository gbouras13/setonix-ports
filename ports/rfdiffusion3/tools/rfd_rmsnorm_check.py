"""Is apex's FusedRMSNorm — the class upstream binds as every RMSNorm of RFdiffusion3 — correct on this GPU?

    python tools/rfd_rmsnorm_check.py [--rows N[,N...]] [--cols C[,C...]] [--json FILE]

apex's layer-norm kernel states its assumption in the source (csrc/layer_norm_cuda_kernel.cu:55): `blockDim.x == warpSize`.
It launches blocks of (32, 4, 1) and reduces each row inside a 32-lane shuffle rotation (`srcLaneB = (threadIdx.x + (1<<l)) & 31`,
`WARP_SHFL(..., srcLaneB)` with the width argument left at its default). On an MI250X a wavefront is 64 lanes, so that default
width is 64: lanes 32-63 of the wavefront — the block's threadIdx.y == 1 row — read the OTHER row's partial sums. The error is
therefore invisible at one row per block and appears as soon as a block holds more than one (the `rows` axis below).

Each case is checked against a float64 CPU reference of the same arithmetic (y = x * rsqrt(mean(x^2) + eps) * w), the rule
upstream's RMSNorm applies. The bar is the CORRECTLY ROUNDED answer IN THE KERNEL'S OWN OUTPUT DTYPE: `cast` below is the
error of the float64 reference rounded to that dtype, i.e. what perfect arithmetic would still cost, and a kernel is right
when it lands within a few multiples of it. Reading apex's bf16 output against torch.nn.RMSNorm's is NOT a fair test and is
not the criterion here: under autocast torch casts `rms_norm` to fp32 and returns fp32, while apex returns bf16, so torch's
column is an order of magnitude tighter for arithmetic reasons alone. torch's number is still printed, as context.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch

EPS = 1e-5
TOL_ULPS = 8          # a kernel may spend a few roundings of its own dtype; 29% (the pinned apex here) is not that


def reference(x: torch.Tensor, w: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    xd = x.detach().to(torch.float64).cpu()
    wd = w.detach().to(torch.float64).cpu()
    return (xd * torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + eps) * wd)


def cast_floor(ref: torch.Tensor, dtype: torch.dtype) -> dict:
    """The error of the float64 reference rounded to `dtype`: the floor any kernel writing that dtype must pay."""
    return err(ref.to(dtype), ref)


def err(got: torch.Tensor, ref: torch.Tensor) -> dict:
    g = got.detach().to(torch.float64).cpu()
    d = (g - ref).abs()
    scale = ref.abs().clamp_min(1e-12)
    return {"max_abs": float(d.max()), "max_rel": float((d / scale).max()), "rms": float(d.pow(2).mean().sqrt())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,2,4,8,64,1024")
    ap.add_argument("--cols", default="64,128,256,384,512,1024")
    ap.add_argument("--dtypes", default="float32,bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    if not torch.cuda.is_available():
        print("no GPU visible", file=sys.stderr)
        return 2
    dev = torch.device("cuda")
    print(f"torch {torch.__version__} hip {torch.version.hip} device {torch.cuda.get_device_name(0)} "
          f"capability {torch.cuda.get_device_capability(0)}")
    try:
        from apex.normalization.fused_layer_norm import FusedRMSNorm
        import importlib.metadata as md
        print(f"apex {md.version('apex')} FusedRMSNorm {FusedRMSNorm.__module__}")
    except Exception as e:  # noqa: BLE001
        print(f"apex not importable: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    rows = [int(v) for v in a.rows.split(",")]
    cols = [int(v) for v in a.cols.split(",")]
    dtypes = [getattr(torch, d) for d in a.dtypes.split(",")]
    results, worst_apex, worst_torch = [], 0.0, 0.0
    torch.manual_seed(a.seed)
    for dt in dtypes:
        for c in cols:
            w = torch.randn(c, device=dev, dtype=torch.float32).abs() + 0.5          # upstream's weights are fp32, and the
            # network calls these modules under bf16 autocast, which is how the bf16 rows below are run
            apex_mod = FusedRMSNorm(c, eps=EPS).to(dev)
            torch_mod = torch.nn.RMSNorm(c, eps=EPS).to(dev)
            with torch.no_grad():
                apex_mod.weight.copy_(w)
                torch_mod.weight.copy_(w)
            for n in rows:
                x = torch.randn(n, c, device=dev, dtype=dt)
                ref = reference(x, w)
                try:
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dt is torch.bfloat16):
                        ya = apex_mod(x)
                        yt = torch_mod(x)
                        ya2 = apex_mod(x)
                except Exception as exc:  # noqa: BLE001 — a cell apex will not serve here is a result, not a crash
                    print(f"{str(dt).split('.')[-1]:>9} rows={n:<5} cols={c:<5} apex RAISED {type(exc).__name__}: {str(exc)[:120]}")
                    results.append({"dtype": str(dt).split(".")[-1], "rows": n, "cols": c,
                                    "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
                    continue
                ea, et = err(ya, ref), err(yt, ref)
                fl = cast_floor(ref, ya.dtype)                      # the bar: perfect arithmetic in apex's OWN output dtype
                det = bool(torch.equal(ya, ya2))
                ok = ea["max_rel"] <= max(TOL_ULPS * fl["max_rel"], 1e-6)
                results.append({"dtype": str(dt).split(".")[-1], "rows": n, "cols": c, "out_dtype": str(ya.dtype),
                                "apex": ea, "torch": et, "cast_floor": fl, "apex_repeatable": det, "ok": ok})
                worst_apex = max(worst_apex, ea["max_rel"] / max(fl["max_rel"], 1e-12))
                worst_torch = max(worst_torch, et["max_rel"])
                print(f"{str(dt).split('.')[-1]:>9} rows={n:<5} cols={c:<5} "
                      f"apex max_rel={ea['max_rel']:.3e} ({ea['max_rel'] / max(fl['max_rel'], 1e-12):.1f}x the "
                      f"{str(ya.dtype).split('.')[-1]} rounding floor) | torch {et['max_rel']:.3e} | "
                      f"repeatable={det}{'' if ok else '   <<< APEX WRONG'}")
    bad = [r for r in results if not r.get("ok", True)]
    print(f"\nWORST apex error: {worst_apex:.1f}x the rounding floor of its own output dtype "
          f"(the bar is {TOL_ULPS}x); torch.nn.RMSNorm's own worst max_rel vs float64: {worst_torch:.3e}")
    verdict = "APEX MATCHES ITS CLASS" if not bad else "APEX IS WRONG ON THIS CARD"
    print(f"VERDICT: {verdict}" + (f" ({len(bad)}/{len(results)} cases over the bar)" if bad else
                                   f" ({len(results)} cases, all within {TOL_ULPS}x the dtype's own rounding)"))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump({"worst_apex_x_floor": worst_apex, "worst_torch_max_rel": worst_torch, "tol_x_floor": TOL_ULPS,
                       "cases_over_bar": len(bad), "verdict": verdict, "cases": results}, fh, indent=1)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
