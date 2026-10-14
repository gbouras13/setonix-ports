"""Run the big line on torch's NATIVE caching allocator (``expandable_segments:False``) instead of the kit's mandatory
expandable segments. Opt-in: ``ODDE_NATIVE_ALLOC=1`` (``patches/hook5``). Separate from v2, the floor and the blocks.

WHY. On ROCm 7.2 (torch 2.14.0+rocm7.2, HIP 7.2.53211), memory released through PyTorch's expandable-segment path stays
counted as used by the process (``slurm/vmmcheck.sbatch``). With live tensors scattered through the segment's range, a
large later request then fails even though it is smaller than what was freed. Filling 10 x (5 + 1) GiB, freeing the 5 GiB
blocks, then asking for one 40 GiB block: OOM under ``expandable_segments:True``, ok under ``:False`` (job 49988904).
That is the structural stage at 10,240 tokens, P=8. ``z_res`` is parked (PyTorch reserved 53.6 -> 13.9 GiB), the driver
still counts 51.8 GiB used, and the 36.57 GiB structural pair shard is refused with 8.62 GiB free. It cannot be switched
mid-run, because memory freed under ``True`` stays stranded. The setting has to be off from the first CUDA call.

WHAT. Two wrappers, because the kit writes the value twice. (1) ``opendde_opt.modes.resolve`` puts the line's own
allocator field (``Line.allocator`` = ``expandable_segments:True``) into ``Resolution.exports`` (modes.py:570), and
activation writes those exports into the environment AFTER the allocator writer below has run. That is why the first
version of this patch, which wrapped only (2), exported ``:False`` and still ran on expandable segments (nativecheck job
49989131: the kit's read-back said ``effective: true``). The wrapper rewrites that one export to ``:False``.
(2) The kit's one allocator writer (``opendde_opt.alloc.apply`` -> ``opt_core.mem.torch_alloc.export``) asks
``torch_alloc.conf_for("expandable")`` for the value it exports, and refuses any other value it finds already present.
Wrapping ``conf_for`` so that ``"expandable"`` yields ``expandable_segments:False`` keeps the writer, its refusals and
its child-environment rows (``env_row``) consistent with each other. Every rank inherits the value before its first CUDA
call. The kit's per-prediction cross-check (``big._prediction``: ``expandable_segments:True`` in
``PYTORCH_CUDA_ALLOC_CONF``) then counts each prediction as ``allocator_absent`` in the census. That is a record, not a
refusal, and it is the honest label for such a run.

COST. The native allocator can fragment where expandable segments would not, and the trunk ran within ~14 GiB of the
card at 10,240 tokens. Numerics are unchanged by construction (placement only), so ``--det 1`` must stay byte-identical.
"""
from __future__ import annotations

import os
import sys

NATIVE = "expandable_segments:False"
_STATE = {"installed": False}


def make_conf_for(orig):
    def conf_for(policy, *args, **kwargs):
        if policy == "expandable":
            return NATIVE
        return orig(policy, *args, **kwargs)
    conf_for._odde_native_alloc = True
    conf_for.__wrapped__ = orig
    return conf_for


ALLOC_VAR = "PYTORCH_CUDA_ALLOC_CONF"
EXPANDABLE = "expandable_segments:True"


def make_resolve(orig):
    def resolve(*args, **kwargs):
        res = orig(*args, **kwargs)
        ex = getattr(res, "exports", None)
        if isinstance(ex, dict) and ex.get(ALLOC_VAR) == EXPANDABLE:
            ex[ALLOC_VAR] = NATIVE
            if not _STATE.get("resolve_said"):
                _STATE["resolve_said"] = True
                line = getattr(getattr(res, "line", None), "name", "?")
                sys.stderr.write("[odde-native-alloc] modes.resolve: line %s exports %s=%s, not %s (pid %d, rank %s)\n"
                                 % (line, ALLOC_VAR, NATIVE, EXPANDABLE, os.getpid(), os.environ.get("ROWPAIR_RANK", "-")))
                sys.stderr.flush()
        return res
    resolve._odde_native_alloc = True
    resolve.__wrapped__ = orig
    return resolve


def make_resolve_logged(orig):
    w = make_resolve(orig)
    sys.stderr.write("[odde-native-alloc] opendde_opt.modes.resolve: the line's allocator export -> %s installed (pid %d, rank %s)\n"
                     % (NATIVE, os.getpid(), os.environ.get("ROWPAIR_RANK", "-")))
    sys.stderr.flush()
    return w


def make_conf_for_logged(orig):
    w = make_conf_for(orig)
    _STATE["installed"] = True
    sys.stderr.write("[odde-native-alloc] opt_core.mem.torch_alloc.conf_for: 'expandable' -> %s installed (pid %d, rank %s)\n"
                     % (NATIVE, os.getpid(), os.environ.get("ROWPAIR_RANK", "-")))
    sys.stderr.flush()
    return w
