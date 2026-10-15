"""The sharded line's diffusion cost per rank: the measurement of the two diffusion words (env-gated engineering diagnostics).

PXS_DIFF_COST=1        every denoiser call of the sharded line (protenix_opt.tp_bind.diffusion._Denoiser.__call__) and its
                       DiffusionTransformer part (opt_core.mem.rowpair.diffusion.diffusion_transformer_sharded) timed with a device
                       sync on either side. The allocator's peak is reset at each call's start and read at its end (allocated and
                       reserved), and the resident allocation and the driver's free memory are read at the first call's start. One line
                       per rank at the end (or at exit): `[pxs-diffcost r<rank>] ...`.
PXS_FAKE_TRUNK=1       the trunk's driver (opt_core.mem.rowpair.trunk.run_trunk_sharded) returns the z_init shard that the kit's own init
                       statement builds (init_pair_shard over the binding's init_rows_fn) and s = the single recycle of zeros: no MSA,
                       template or Pairformer. The diffusion's inputs keep their shapes, dtypes and sharding; their values are not a
                       trunk's. For cost only: a structure from such a run means nothing.
PXS_DIFF_EXIT_AFTER=k  after the k-th denoiser call every rank prints its line, meets the others at a barrier and ends (os._exit(0)): the
                       confidence head is not run.
Armed at interpreter start from setonix_rankcache.py, in torchrun workers only (WORLD_SIZE > 1). The kit's route and bind hooks sit on
ptx_tp.* modules; the objects patched here are the core's and the binding's own module attributes, looked up at call time.
"""
import atexit
import os
import statistics
import sys
import time

S = {"calls": [], "dit": [], "peak_alloc": 0, "peak_res": 0, "base_alloc": None, "base_res": None, "base_free": None,
     "reported": False, "fake_trunk": False, "fake_shape": None}


def _rank():
    return os.environ.get("RANK") or os.environ.get("ROWPAIR_RANK") or "0"


def say(msg):
    print(f"[pxs-diffcost r{_rank()}] {msg}", file=sys.stderr, flush=True)


def _gib(b):
    return float("nan") if b is None else b / 2 ** 30


def _med(v):
    return statistics.median(v) if v else float("nan")


def words():
    return (f"dit_rows={os.environ.get('ROWPAIR_DIT_ROWS') or 'big(default)'} "
            f"bias_cache_dtype={os.environ.get('ROWPAIR_DIFF_BIAS_CACHE_DTYPE') or 'fp16(default)'}")


def report():
    if S["reported"]:
        return
    S["reported"] = True
    c, d = S["calls"], S["dit"]
    if not c:
        say(f"no denoiser call recorded; {words()} fake_trunk={int(S['fake_trunk'])}")
        return
    say(f"{words()} fake_trunk={int(S['fake_trunk'])} calls={len(c)} first_call_s={c[0]:.2f} rest_median_s={_med(c[1:]):.2f} "
        f"dit_first_s={d[0] if d else float('nan'):.2f} dit_rest_median_s={_med(d[1:]):.2f} denoise_total_s={sum(c):.1f} "
        f"base_alloc_gib={_gib(S['base_alloc']):.2f} base_reserved_gib={_gib(S['base_res']):.2f} base_free_gib={_gib(S['base_free']):.2f} "
        f"peak_alloc_gib={_gib(S['peak_alloc']):.2f} peak_reserved_gib={_gib(S['peak_res']):.2f} "
        f"calls_s={[round(x, 2) for x in c]} dit_s={[round(x, 2) for x in d]}")


def arm_fake_trunk():
    from opt_core.mem.rowpair import trunk as CT
    orig = CT.run_trunk_sharded

    def run_trunk_sharded(layout, *, n_cycles, init_rows_fn, init_like, recycle_update_fn=None, s_init=None, single_recycle_fn=None,
                          init_rows=None, **_kw):
        import torch
        t0 = time.time()
        z_loc = CT.init_pair_shard(layout, init_rows_fn, init_like, init_rows)
        s = single_recycle_fn(torch.zeros_like(s_init), 0) if (single_recycle_fn is not None and s_init is not None) else s_init
        S["fake_shape"] = (tuple(z_loc.shape), str(z_loc.dtype))
        say(f"FAKE TRUNK: z = the z_init shard {tuple(z_loc.shape)} {str(z_loc.dtype).replace('torch.', '')} in {time.time() - t0:.1f} s; "
            f"s = the single recycle of zeros; no MSA / template / Pairformer ({n_cycles} cycle(s) not run)")
        rec = {"timings_s": {"fake_trunk_init": round(time.time() - t0, 3)}, "park": "none", "fake_trunk": True}
        return CT.TrunkOut(s, z_loc, layout, True, 0, False, rec)
    run_trunk_sharded._pxs_original = orig
    CT.run_trunk_sharded = run_trunk_sharded
    S["fake_trunk"] = True


def arm(force=False):
    if not force and (os.environ.get("WORLD_SIZE") or "1").strip() in ("", "0", "1"):
        return False
    import torch
    from opt_core.mem.rowpair import diffusion as CD
    from protenix_opt.tp_bind import diffusion as TD
    cuda = torch.cuda.is_available()

    def sync():
        if cuda:
            torch.cuda.synchronize()

    orig_call = TD._Denoiser.__call__

    def __call__(self, *a, **k):
        sync()
        if cuda:
            if S["base_alloc"] is None:
                S["base_alloc"], S["base_res"] = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
                S["base_free"] = torch.cuda.mem_get_info()[0]
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        out = orig_call(self, *a, **k)
        sync()
        S["calls"].append(time.time() - t0)
        if cuda:
            S["peak_alloc"] = max(S["peak_alloc"], torch.cuda.max_memory_allocated())
            S["peak_res"] = max(S["peak_res"], torch.cuda.max_memory_reserved())
        k_exit = int(os.environ.get("PXS_DIFF_EXIT_AFTER") or 0)
        if k_exit and len(S["calls"]) >= k_exit:
            report()
            try:
                import torch.distributed as dist
                if dist.is_initialized():
                    dist.barrier()
            except Exception:  # noqa: BLE001 — the measurement is printed; the exit must not hang on a broken group
                pass
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        return out
    TD._Denoiser.__call__ = __call__

    orig_dit = CD.diffusion_transformer_sharded

    def diffusion_transformer_sharded(*a, **k):
        sync()
        t0 = time.time()
        out = orig_dit(*a, **k)
        sync()
        S["dit"].append(time.time() - t0)
        return out
    CD.diffusion_transformer_sharded = diffusion_transformer_sharded

    if os.environ.get("PXS_FAKE_TRUNK") == "1":
        arm_fake_trunk()
    atexit.register(report)
    say(f"armed: {words()} fake_trunk={int(S['fake_trunk'])} exit_after={os.environ.get('PXS_DIFF_EXIT_AFTER') or 'never'} "
        f"HIP_LAUNCH_BLOCKING={os.environ.get('HIP_LAUNCH_BLOCKING')} alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
    return True
