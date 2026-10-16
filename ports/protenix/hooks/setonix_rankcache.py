"""Per-rank JIT cache directories for the Protenix v2 kit's sharded line on Setonix.

The kit's `--n_gpu P` launcher is torchrun (ptx_tp.launch -> torch.distributed.run), and
every worker inherits one environment, so all P ranks would share one TRITON_CACHE_DIR,
one MIOpen database and one TorchInductor cache. Concurrent writers to one Triton cache
is a known corruption source on this machine.

Loaded at interpreter start by setonix_rankcache.pth in the venv's site-packages. It only
acts inside a multi-rank worker (torchrun's LOCAL_RANK with WORLD_SIZE > 1, or opt_core's
ROWPAIR_RANK) and only rewrites cache variables that are already set. It imports
nothing heavy and touches nothing else.
"""
import os


def _apply():
    env = os.environ
    rank = None
    if (env.get("WORLD_SIZE") or "1").strip() not in ("", "0", "1"):
        rank = env.get("LOCAL_RANK") or env.get("RANK")
    if rank is None:
        rank = env.get("ROWPAIR_RANK")
    if rank is None or not rank.strip().isdigit():
        return
    for var in ("TRITON_CACHE_DIR", "MIOPEN_USER_DB_PATH", "MIOPEN_CUSTOM_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        base = env.get(var)
        if not base or os.path.basename(base.rstrip("/")) == "rank" + rank.strip():
            continue
        path = os.path.join(base, "rank" + rank.strip())
        try:
            os.makedirs(path, exist_ok=True)
        except OSError:
            continue
        env[var] = path


_apply()

if os.environ.get("PXS_NAN_PROBE") == "1":        # diagnostics only: global module hooks (setonix_nanprobe.py)
    try:
        import setonix_nanprobe
        setonix_nanprobe.arm()
    except Exception as _e:                        # never break an interpreter over a diagnostic
        import sys
        print(f"[nan_probe] not armed: {_e!r}", file=sys.stderr, flush=True)

if os.environ.get("PXS_TRIATT_CHUNK_CAP"):       # capacity lever: stock's triangle-attention chunk capped (setonix_chunkcap.py)
    try:
        import setonix_chunkcap
        setonix_chunkcap.arm()
    except Exception as _e:
        import sys
        print(f"[pxs-chunkcap] not armed: {_e!r}", file=sys.stderr, flush=True)

if os.environ.get("PXS_DETERMINISTIC") == "1" or os.environ.get("PXS_DUMP_PRED"):   # diagnostics: deterministic algorithms, prediction dumps (setonix_det.py)
    try:
        import setonix_det
        setonix_det.arm()
    except Exception as _e:
        import sys
        print(f"[pxs-det] not armed: {_e!r}", file=sys.stderr, flush=True)

if os.environ.get("PXS_DIFF_COST") == "1":        # diagnostics: the sharded line's diffusion cost per rank, optional fake trunk (setonix_diffcost.py)
    try:
        import setonix_diffcost
        setonix_diffcost.arm()
    except Exception as _e:
        import sys
        print(f"[pxs-diffcost] not armed: {_e!r}", file=sys.stderr, flush=True)
