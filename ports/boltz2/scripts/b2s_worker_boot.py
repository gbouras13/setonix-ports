#!/usr/bin/env python3
"""b2s_worker_boot.py -- start a Boltz-2 kit worker (``python -m boltz2_opt.worker_launch ...``) after installing opt-in, worker-side
patches for gfx90a. OPT-IN and default off: nothing here runs unless scripts/b2s_launch.py was given B2S_WORKER_PATCHES, in which case it
rewrites the kit's worker command line (stack.worker_command: the single-GPU worker and every rank of --n_gpu P) to

    python b2s_worker_boot.py -m boltz2_opt.worker_launch <the kit's own arguments, unchanged>

The worker then runs exactly as the kit launched it (runpy, run_name="__main__"). The patches refuse opt_core rows by name, at a point
where the provider's own logic takes over: the TriMul's row admission (its cell walk moves on), the transition's serve step (its
inherited-row loop steps aside to the next row). The census records every refusal. No file of the kit or the core is changed (the kit
digests its files).

B2S_WORKER_PATCHES (comma list):
  transition_no_v2   opt_core's transition provider refuses to SERVE row `v2` (opt_core fpf_transition_v2) on gfx90a. That row serves
                     a few calls above 256 tokens (the hidden-256 cell's measured launch word v2@...) and returns all-NaN there
                     (smokes 1-4; Protenix's `transition_core`, the same kernel, was all-NaN too). The refusal kind starts `no_cell:`,
                     the kit's declared word for "no cell for this call shape" (here: none valid on this card), in case it escapes to
                     the kit's binder, which then serves the call with the module's own forward and accepts the word.
                     On this stack (unmeasured, inherited from the H100 cells) the provider's transition() loop catches the refusal,
                     marks v2 dead for the cell and re-selects its next portable row, af3_fused where measured: the same path a row
                     that fails to build takes. History: smoke 10 (job 50379835) refused v2 in admits(), valid 2 (job 50387326) in
                     _select(); both pre-empted that loop, so every call took the module's forward (finite, rc 0 in valid 2, but
                     af3_fused never served).
  trimul_v4_only     opt_core's TriMul row admission (opt_core.kernels.trimul.admits) refuses every kernel row except `v4` on gfx90a
                     (the stock `cueq` / `torch_math` statements stay admitted): the cell's row walk then lands on v4, opt_core
                     fpf_trimul_v4, no inline asm, within stock's precision against fp64 on gfx90a in the Protenix port. Smoke 10: with
                     only `native` refused, fast's cell for N <= 400 walked on to `esm_v5_fwd` (NVPTX inline asm) and the run was all-NaN.
  trimul_no_native   an alias of trimul_v4_only (the first, narrower version's name).
  alloc_no_expandable  sets PYTORCH_CUDA_ALLOC_CONF to expandable_segments:False in the worker / rank process at boot, before torch
                     is imported (other keys kept). big's memory line always applies the core's expandable_segments lever (big.py
                     row_settings keeps it whatever the row says: a card drop of the lever only removes it from the parent's list, and
                     the worker then sets it through the runtime API; the parent's gate names that "applied outside the row", rc 3,
                     ladder 2026-10-05). opt_core.mem.allocator's own `conf` precondition treats a pre-set expandable_segments:False
                     as the kit's pin and never overrides it: the lever refuses by name, the allocator stays on torch's default
                     (non-expandable) segments, and the parent's gate names the refusal (rc 3, or --allow-partial). For A/B
                     capacity runs: expandable_segments:True stranded memory on ROCm 7.2 in the Protenix port.
                     MEASURED 2026-10-05 (ladderc 50421726): the refusal aborts big's WHOLE memory attachment ("REFUSED xl"), the
                     worker never runs (rc 1). Kept for the record; use xl_no_expandable.
  xl_no_expandable   composes big's memory line WITHOUT the expandable_segments lever: boltz2_opt.big.row_settings (the row read into
                     the line's levers) returns its row minus that lever, so the line the core applies never touches the allocator
                     (torch's default, non-expandable segments). Pair it with the parent-side card drop
                     "big:expandable_segments:<why>", so the parent's row and the worker's line agree (no "applied outside the
                     row"). Applies in the single-GPU worker and in every --n_gpu rank.
  triattn_tiles      sets PTX_TRIATTN_CONFIG (the tile word opt_core's flash triangle-attention kernels read ONCE at import:
                     flash_triattn.py, fpf_triatt_k2b/{triatt_k2b,flash_triattn_k2}.py, "BLOCK_M,BLOCK_N,ROWS,num_warps,num_stages[,ORDER]",
                     overriding their H100-tuned tables) to B2S_TRIATTN_CONFIG, default 128,32,2,4,1,1: the OpenFold3 port's
                     gfx90a retune of the same kernel (bitwise there; 6,930-token block 5.67 -> 3.48 s). The kit strips PTX_TRIATTN_*
                     from the worker environment (stock/PINS.json must_be_absent_prefixes), so it is set here, inside the worker and
                     every rank, before anything imports the kernel.
  pairblock_kernels  lets the kit's fused triangle-attention block (boltz2_opt.pairblock) serve under the kernels-off route when its
                     variant's attention core is opt_core's own (BOLTZ_PAIRBLOCK=flash | k2b): the block's gate is boltz's own
                     use_kernels flag, and the port must run kernels off (no cuEquivariance on ROCm), so the block never served. The
                     block's apply() is wrapped: before it, TriangleAttention.forward becomes stock's forward with use_kernels forced
                     False (the block keeps the forward it finds as its fallback, so every declined call runs stock's code exactly as
                     now); after it, one outer forward hands the block use_kernels=True. Any other variant is left alone. Added for
                     fast on one GCD by b2s_launch.py's fast flash default (2026-10-12; B2S_FAST_FLASH=0 opts out). Measured
                     (fastflash 50659670): 2.89x stock with the row patch, deterministic, 0.13 A from the big default.
Every refusal word contains `b2s` so it can be read back from the worker log's census lines.
"""
from __future__ import annotations

import importlib.abc
import os
import runpy
import sys

PATCHES = tuple(p.strip() for p in os.environ.get("B2S_WORKER_PATCHES", "").split(",") if p.strip())
KNOWN = ("transition_no_v2", "trimul_v4_only", "trimul_no_native", "alloc_no_expandable", "xl_no_expandable", "triattn_tiles",
         "pairblock_kernels")
BOOT_ONLY = ("alloc_no_expandable", "triattn_tiles")  # act on the environment at boot, no module hook
TORCH_CONF = "PYTORCH_CUDA_ALLOC_CONF"
TRIATTN_TILES_DEFAULT = "128,32,2,4,1,1"
_ARCH = {"v": None}


def _say(msg: str) -> None:
    print(f"[b2s-boot pid={os.getpid()}] {msg}", file=sys.stderr, flush=True)


def _on_gfx90a() -> bool:
    """True when the node's GPUs are AMD gfx90a (KFD topology: gfx_target_version 90010), read from sysfs so that no CUDA context is
    created as a side effect (the kit tracks which of its processes fork before CUDA). Cached; False when there is no KFD topology."""
    if _ARCH["v"] is None:
        found = False
        topo = "/sys/class/kfd/kfd/topology/nodes"
        try:
            for n in os.listdir(topo):
                try:
                    with open(os.path.join(topo, n, "properties")) as fh:
                        if any(ln.split()[:2] == ["gfx_target_version", "90010"] for ln in fh):
                            found = True
                            break
                except OSError:
                    continue
        except OSError:
            found = False
        _ARCH["v"] = found
    return _ARCH["v"]


V2_KIND = "no_cell:b2s_gfx90a_v2_all_nan(fpf_transition_v2)"


def _patch_transition(mod) -> None:
    orig_serve = mod._serve_v2

    def _serve_v2(x, W, sel, *a, **kw):
        # Refused at SERVE time only. On this (unmeasured, inherited) stack the provider's transition() loop catches a refusal, marks the
        # row dead for the cell and re-selects its next portable row (af3_fused where the cell measured it), exactly as it does when a
        # row fails to build. A refusal that escapes (a non-inherited path) carries the kit's declared word `no_cell`.
        if _on_gfx90a():
            raise mod.Refusal(V2_KIND, getattr(sel, "word", None), "torch_swiglu")
        return orig_serve(x, W, sel, *a, **kw)

    _serve_v2.__wrapped__ = orig_serve
    mod._serve_v2 = _serve_v2
    _say("patch transition_no_v2: opt_core.kernels.transition refuses to SERVE row v2 on gfx90a (the provider steps aside to its next row)")


TRIMUL_ADMITTED = ("v4", "cueq", "torch_math")


def _patch_trimul(mod) -> None:
    orig = mod.admits

    def admits(row, *a, **kw):
        if row not in TRIMUL_ADMITTED and _on_gfx90a():
            raise mod.Refusal("b2s:gfx90a_row_unvalidated:%s" % row, row, "v4")
        return orig(row, *a, **kw)

    admits.__wrapped__ = orig
    mod.admits = admits
    _say("patch trimul_v4_only: opt_core.kernels.trimul admits only v4 (and the stock statements) on gfx90a")


def _pin_no_expandable() -> None:
    before = os.environ.get(TORCH_CONF)
    keys = [kv.strip() for kv in (before or "").split(",") if kv.strip() and not kv.strip().startswith("expandable_segments:")]
    os.environ[TORCH_CONF] = ",".join(keys + ["expandable_segments:False"])
    _say(f"pin alloc_no_expandable: {TORCH_CONF}={os.environ[TORCH_CONF]!r} (was {before!r}); opt_core's expandable_segments lever "
         f"refuses a pre-set False by its own conf precondition")


def _patch_big(mod) -> None:
    orig = mod.row_settings

    def row_settings(env):
        row = orig(env)
        if row is not None and "expandable_segments" in tuple(row.get("levers") or ()):
            row = dict(row)
            row["levers"] = tuple(lv for lv in row["levers"] if lv != "expandable_segments")
            _say(f"xl_no_expandable: big's memory line composed without expandable_segments (levers {','.join(row['levers'])})")
        return row

    row_settings.__wrapped__ = orig
    mod.row_settings = row_settings
    _say("patch xl_no_expandable: boltz2_opt.big.row_settings leaves expandable_segments out of the memory line")


def _set_triattn_tiles() -> None:
    word = os.environ.get("B2S_TRIATTN_CONFIG", TRIATTN_TILES_DEFAULT).strip()
    vals = [v.strip() for v in word.split(",")]
    if len(vals) not in (5, 6) or not all(v.isdigit() for v in vals):
        raise SystemExit(f"[b2s-boot] B2S_TRIATTN_CONFIG must be BLOCK_M,BLOCK_N,ROWS,num_warps,num_stages[,ORDER], not {word!r}")
    before = os.environ.get("PTX_TRIATTN_CONFIG")
    os.environ["PTX_TRIATTN_CONFIG"] = ",".join(vals)
    _say(f"triattn_tiles: PTX_TRIATTN_CONFIG={os.environ['PTX_TRIATTN_CONFIG']} (was {before!r}); opt_core's flash triangle-attention "
         f"kernels read it at import")


PAIRBLOCK_OWN_CORES = ("flash", "k2b")       # the pair block's variants whose attention core is opt_core's own kernel (no cuEquivariance)


def _patch_pairblock(mod) -> None:
    orig_apply = mod.apply

    def apply(spec=None):
        import boltz.model.layers.triangular_attention.attention as A
        stock = A.TriangleAttention.forward
        if not getattr(stock, "_b2s_kernels_off", False):
            def stock_kernels_off(self, x, mask=None, chunk_size=None, use_kernels=False):
                # Every call the kit's block does not serve lands here (the block keeps the forward it found as its fallback) and runs
                # stock's own code exactly as the kernels-off route does: cuEquivariance is never asked for.
                return stock(self, x, mask=mask, chunk_size=chunk_size, use_kernels=False)

            stock_kernels_off._b2s_kernels_off = True
            stock_kernels_off.__wrapped__ = stock
            A.TriangleAttention.forward = stock_kernels_off
        out = orig_apply(spec)
        block = A.TriangleAttention.forward
        v = (getattr(mod, "_STATE", None) or {}).get("variant")
        if getattr(block, "_b2s_kernels_off", False) or v not in PAIRBLOCK_OWN_CORES:
            _say(f"pairblock_kernels: the kit's block is {'not applied' if getattr(block, '_b2s_kernels_off', False) else 'variant ' + str(v)}; "
                 f"left on the kernels-off route (serves only with an opt_core core: {','.join(PAIRBLOCK_OWN_CORES)})")
            return out

        def forward(self, x, mask=None, chunk_size=None, use_kernels=False):
            # The kit's block serves only when boltz asks for kernels (its gate is boltz's own use_kernels flag), and the port runs
            # kernels off because cuEquivariance does not exist on ROCm. With an opt_core core the block needs no cuEquivariance, so
            # it is asked to serve; whatever it declines reaches stock_kernels_off above.
            return block(self, x, mask, chunk_size, True)

        forward.__wrapped__ = block
        A.TriangleAttention.forward = forward
        _say(f"pairblock_kernels: the kit's fused triangle-attention block (variant {v}: opt_core's own attention core) serves under the "
             f"kernels-off route; every call it declines runs stock's code with use_kernels=False")
        return out

    apply.__wrapped__ = orig_apply
    mod.apply = apply
    _say("patch pairblock_kernels: boltz2_opt.pairblock.apply wrapped (takes effect when the kit attaches its block)")


HOOKS = {"transition_no_v2": ("opt_core.kernels.transition", _patch_transition),
         "trimul_v4_only": ("opt_core.kernels.trimul", _patch_trimul),
         "trimul_no_native": ("opt_core.kernels.trimul", _patch_trimul),
         "xl_no_expandable": ("boltz2_opt.big", _patch_big),
         "pairblock_kernels": ("boltz2_opt.pairblock", _patch_pairblock)}


class _PostImport(importlib.abc.MetaPathFinder):
    """Runs a patch right after the named module executes, wherever and whenever the worker first imports it (import order unchanged).
    The hook is consumed when the module EXECUTES, not when it is looked up: the kit locates some of its own modules with
    importlib.util.find_spec (to digest the file) before importing them, and each lookup builds a fresh spec and loader (measured
    2026-10-05: boltz2_opt.big's hook was spent on such a lookup and never ran, ladderc 50436697)."""

    def __init__(self, hooks):
        self.hooks = dict(hooks)

    def find_spec(self, fullname, path, target=None):
        if fullname not in self.hooks:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        loader, hook = spec.loader, self.hooks[fullname]
        if loader is None or not hasattr(loader, "exec_module"):
            _say(f"cannot patch {fullname}: loader {loader!r} has no exec_module")
            return spec
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            if self.hooks.pop(fullname, None) is not None:   # the first execution only
                hook(module)

        loader.exec_module = exec_module
        return spec


def install() -> None:
    unknown = [p for p in PATCHES if p not in KNOWN]
    if unknown:
        raise SystemExit(f"[b2s-boot] unknown B2S_WORKER_PATCHES word(s): {','.join(unknown)} (known: {','.join(KNOWN)})")
    if "triattn_tiles" in PATCHES:
        hit = [m for m in sys.modules if m.startswith("opt_core.kernels.") and "triat" in m]
        if hit:                                       # not expected: the kernels read the word at import
            _say(f"triattn_tiles: already imported at boot ({','.join(hit)}); the tiles may come too late")
        _set_triattn_tiles()
    if "alloc_no_expandable" in PATCHES:
        if "torch" in sys.modules:                    # not expected: the allocator may already have read the variable
            _say("alloc_no_expandable: torch already imported at boot; the pin may come too late")
        _pin_no_expandable()
    hooks = {}
    for p in dict.fromkeys(PATCHES):                   # the alias and its target patch the same module once
        if p in BOOT_ONLY:
            continue
        name, fn = HOOKS[p]
        if name in hooks:
            continue
        if name in sys.modules:                       # already imported (not expected this early): patch in place
            fn(sys.modules[name])
        else:
            hooks[name] = fn
    if hooks:
        sys.meta_path.insert(0, _PostImport(hooks))
    _say(f"patches={','.join(PATCHES) or '-'}")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "-m":
        raise SystemExit("usage: b2s_worker_boot.py -m <module> [args ...]")
    install()
    module = sys.argv[2]
    sys.argv = [module] + sys.argv[3:]
    # `python -m` puts the current directory first on sys.path; `python <this file>` put this file's directory there instead. The
    # kit stages its worker files in the working directory and imports them through it, so restore exactly what -m would give.
    here = os.path.dirname(os.path.abspath(__file__))
    if sys.path and os.path.abspath(sys.path[0] or ".") == here:
        sys.path[0] = os.getcwd()
    else:
        sys.path.insert(0, os.getcwd())
    runpy.run_module(module, run_name="__main__", alter_sys=True)
