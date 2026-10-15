"""NaN/Inf birthplace probe, armed in any interpreter (single process or a torchrun rank).

Uses torch's GLOBAL module hooks (torch.nn.modules.module.register_module_forward_[pre_]hook), so
no kit patch site is touched (the kit patches InferenceRunner.predict / load_checkpoint through
opt_core.autoload, one patch per site). A post-hook fires after its module's forward, so children
report before parents: the first record whose inputs were all finite is where a non-finite value
is born. Qualified names come from the root `Protenix` module's pre-hook.

Armed by setonix_rankcache.py when PXS_NAN_PROBE=1. PXS_NAN_MAXLOG: records printed (default 12).
"""
import atexit
import os
import sys

_S = {"names": {}, "n": 0, "first": None, "calls": 0}
MAXLOG = int(os.environ.get("PXS_NAN_MAXLOG", "12"))
RANK = os.environ.get("RANK") or os.environ.get("ROWPAIR_RANK") or "0"


def _finite(x):
    # reductions only: amax/amin propagate NaN and expose +-inf in O(1) extra memory. A full
    # torch.isfinite(x) mask of an [N, N, 1024] activation is GiBs and faked an OOM (nan2 K3).
    import torch
    if torch.is_tensor(x):
        if not (x.is_floating_point() and x.numel()):
            return True
        return bool(torch.isfinite(x.amax()) and torch.isfinite(x.amin()))
    if isinstance(x, (list, tuple)):
        return all(_finite(v) for v in x)
    if isinstance(x, dict):
        return all(_finite(v) for v in x.values())
    return True


def _desc(x, depth=0):
    import torch
    if torch.is_tensor(x):
        s = f"{tuple(x.shape)}/{str(x.dtype).replace('torch.', '')}"
        if x.is_floating_point() and x.numel():
            s += f"(amax={float(x.amax()):.3g},amin={float(x.amin()):.3g})"   # reductions only (see _finite)
        return s
    if depth > 1:
        return type(x).__name__
    if isinstance(x, (list, tuple)):
        return "[" + ",".join(_desc(v, depth + 1) for v in list(x)[:4]) + "]"
    if isinstance(x, dict):
        return "{" + ",".join(f"{k}:{_desc(v, depth + 1)}" for k, v in list(x.items())[:6]) + "}"
    return type(x).__name__


def _pre(mod, args):
    if not _S["names"] and type(mod).__name__ == "Protenix":
        _S["names"] = {m: n for n, m in mod.named_modules()}
        print(f"[nan_probe r{RANK}] armed: {len(_S['names'])} named modules under Protenix", file=sys.stderr, flush=True)


def _post(mod, args, out):
    _S["calls"] += 1
    if _S["n"] >= MAXLOG or not _S["names"] or _finite(out):
        return
    ok = _finite(args)
    name = _S["names"].get(mod, "<" + type(mod).__name__ + ">")
    _S["n"] += 1
    if ok and _S["first"] is None:
        _S["first"] = name
    print(f"[nan_probe r{RANK}] #{_S['n']} {'BORN-HERE' if ok else 'propagated'} module={name} cls={type(mod).__name__} "
          f"out={_desc(out)} args={_desc(args)}", file=sys.stderr, flush=True)


def _summary():
    print(f"[nan_probe r{RANK}] SUMMARY hook_calls={_S['calls']} nonfinite_records={_S['n']} first_born={_S['first']}", file=sys.stderr, flush=True)


def arm():
    from torch.nn.modules import module as M
    M.register_module_forward_pre_hook(_pre)
    M.register_module_forward_hook(_post)
    atexit.register(_summary)
