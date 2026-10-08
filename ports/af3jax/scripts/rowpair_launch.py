#!/usr/bin/env python3
"""rowpair_launch.py SCRIPT [FLAGS...] -- stock run_alphafold.py, row-sharded over the GCDs of one node (af3jax-setonix).

The kit's multi-GPU path (af3_jax_opt/big_launch.py, ``--mode big --n_gpu P``) first installs its NVIDIA-tuned base (FlashPairformer,
the Pallas GLUT/ATTNCFG levers, the memory levers), whose kernels are tiled past gfx90a's 64 KiB LDS. Under P > 1 the kit itself keeps
only one memory lever and hands everything else to the row-sharded recipe (modes.ROWPAIR_SUPERSEDES), so this launcher installs exactly
that and nothing NVIDIA-shaped:

  1. transition_shard: ``Model.Config.global_config.pair_transition_shard_spec = ((None, ROWS),)`` -- the pair transitions in ROWS-row
     sub-batches (a stock AF3 knob, set the kit's way through ``opt_core.mem.jax_mem.set_config_path``; ROWS = the kit's 256);
  2. ``inprocess/rowpair.py``: the mesh over the first P devices and the shared AlphaFold 3 recipe (``opt_core.mem.rowpair_jax.alphafold3``,
     whose eleven pinned source hashes match this stack byte for byte), with ``b21 = big_levers`` (the kit's diffusion pair-conditioning
     transcription; stdlib + opt_core at import);
  3. the runner bound to the mesh -- the kit's runner through ``rowpair.patch_runner``; the stock ``run_alphafold.py`` (which jits in
     ``ModelRunner._model``, not the kit's ``_jitted_apply``) through :func:`bind_stock_runner`, the same rebinding -- then its ``main``
     under absl with the caller's flags. Pass no ``--cache_dir``: it would start tokamax autotuning under the mesh.

Environment: AF3_JAX_N_GPU = P (2, 4 or 8; default 8), ROWPAIR_SCHEDULE = ring | gather (default ring, the memory-lean one: gather
all-gathers the partner plane), ROWPAIR_TRANSITION_ROWS (default 256), AF3JAX_KIT (the kit's af3_jax/opt/af3_jax_opt), AF3JAX_CORE
(common/opt_core). Needs --xla_gpu_unsupported_use_all_reduce_one_shot_kernel=false in XLA_FLAGS on ROCm (small psums hang without it).

ROWPAIR_TRIATTN=flash (C2e): triangle attention through tokamax's Pallas-Triton flash attention with a gfx90a tile (ROWPAIR_TRIATTN_CFG,
default 128:64:4:1), by perf_launch.py's own pin_triattn, loaded by path from this directory (the lever validated on one GCD, not a copy):
tokamax's 'triton' slot rebound with the pinned op, that op alone told it is supported on ROCm, --flash_attention_implementation=triton
appended (absl keeps the last). The recipe's sharded GridSelfAttention body calls the stock _attention on each GCD's rows
(opt_core.mem.rowpair_jax.alphafold3), so the pin reaches every row block; the call report prints at exit. Leave TOKAMAX_ROCM_TRITON unset.

Per-device allocator peaks print from a ``finally`` block while the backend is still up (run_with_peak.py's exit handler ran after jax's
own shutdown and read a fresh allocator), with this process's peak host memory (``[host] maxrss_GiB``). The kit's ROWPAIR install line and
LEVER exit line print as in the kit.
"""
import importlib.util
import os
import sys
import time

T0 = time.time()
TAG = "[rowpair-launch]"


def host_peak() -> None:
    try:
        import resource
        unit = 2**30 if sys.platform == "darwin" else 2**20                # ru_maxrss: bytes on macOS, KiB on Linux
        print(f"[host] maxrss_GiB={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / unit:.2f} wall_s={time.time() - T0:.0f}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[host] maxrss unavailable: {type(e).__name__}: {e}", flush=True)


def pin_flash(cfg: str):
    """perf_launch.py's pin_triattn (loaded by path: the code validated on one GCD, C2), returning (reporter, tile label)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "perf_launch.py")
    spec = importlib.util.spec_from_file_location("af3jax_perf_launch", path)
    pl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pl)                                         # its main() is behind the __main__ guard
    return pl.pin_triattn(cfg, "triton")


def peaks(tag: str = "[peak]") -> None:
    try:
        import jax
        for d in jax.local_devices():
            st = d.memory_stats() or {}
            pk, lim = st.get("peak_bytes_in_use"), st.get("bytes_limit")
            gib = lambda b: (b / 2**30) if isinstance(b, int) else float("nan")  # noqa: E731
            print(f"{tag} device={d.id} peak_bytes_in_use={pk} bytes_limit={lim} peak_GiB={gib(pk):.2f} "
                  f"limit_GiB={gib(lim):.2f} wall_s={time.time() - T0:.0f}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"{tag} unavailable: {type(e).__name__}: {e}", flush=True)


def install_transition_shard(rows: int) -> None:
    from alphafold3.model import model as af3_model
    from opt_core.mem.jax_mem import set_config_path
    stock_init = af3_model.Model.__init__

    def __init__(self, config, *args, **kwargs):
        rec = set_config_path(config, "global_config.pair_transition_shard_spec", ((None, rows),))
        print(f"{TAG} transition_shard pair_transition_shard_spec {rec['stock_value']!r} -> {rec['value']!r}", flush=True)
        return stock_init(self, config, *args, **kwargs)

    __init__.__wrapped__ = stock_init
    af3_model.Model.__init__ = __init__


def bind_stock_runner(mod, rmesh) -> None:
    """The stock runner's equivalent of ``rowpair.patch_runner`` (which binds the kit's runner: ``ModelRunner._jitted_apply``). The stock
    ``run_alphafold.py`` jits in ``ModelRunner._model`` (a cached property returning ``partial(jit(apply, device=), params)``) and places the
    batch with ``jax.device_put(..., self._device)`` in ``run_inference``. Rebound here exactly as the kit rebinds its own runner:
    ``_model`` = ``partial(haiku.jit_apply(apply, rmesh), params)`` (replicated in/out; the pair is sharded inside the program) and
    ``run_inference`` with its one placement call redirected to the mesh, replicated."""
    import functools
    from opt_core.mem.rowpair_jax import haiku as rp_hk, shard
    MR, hk, model, jax = mod.ModelRunner, mod.hk, mod.model, mod.jax

    def _model(self):
        @hk.transform
        def forward_fn(batch):
            return model.Model(self._model_config)(batch)
        apply_fn = forward_fn.apply
        if not mod._NOJIT.value:
            apply_fn = rp_hk.jit_apply(apply_fn, rmesh)
        return functools.partial(apply_fn, self.model_params)

    prop = functools.cached_property(_model)
    prop.__set_name__(MR, "_model")
    MR._model = prop
    stock_run_inference = MR.run_inference

    def run_inference(self, featurised_example, rng_key):
        orig = jax.device_put
        target = shard.named(rmesh, shard.replicated_spec())

        def _put(x, device=None, *a, **k):  # the runner's one placement call, onto the mesh (the ORIGINAL device_put underneath)
            return orig(x, target)
        jax.device_put = _put
        try:
            return stock_run_inference(self, featurised_example, rng_key)
        finally:
            jax.device_put = orig

    MR.run_inference = run_inference
    print(f"{TAG} stock runner bound: ModelRunner._model=partial(jit_apply(mesh), params) run_inference=device_put(mesh, replicated)", flush=True)


def main() -> None:
    if len(sys.argv) < 2 or not sys.argv[1].endswith(".py"):
        sys.exit("usage: rowpair_launch.py SCRIPT.py [flags...]")
    script, flags = os.path.abspath(sys.argv[1]), sys.argv[2:]
    kit, core = os.environ["AF3JAX_KIT"], os.environ["AF3JAX_CORE"]
    n = int(os.environ.get("AF3_JAX_N_GPU", "8"))
    schedule = os.environ.get("ROWPAIR_SCHEDULE", "ring")
    rows = int(os.environ.get("ROWPAIR_TRANSITION_ROWS", "256"))
    triattn = os.environ.get("ROWPAIR_TRIATTN", "")
    if triattn not in ("", "flash"):
        sys.exit(f"{TAG} ROWPAIR_TRIATTN={triattn!r}: 'flash' or unset")
    for p in (core, kit, os.path.dirname(script)):
        sys.path.insert(0, p)
    reporter, tile = None, "stock"
    if triattn == "flash":                                              # before the recipe and the runner: nothing is traced yet
        reporter, tile = pin_flash(os.environ.get("ROWPAIR_TRIATTN_CFG", "128:64:4:1"))
        flags = list(flags) + ["--flash_attention_implementation=triton"]
    import big_levers  # noqa: E402  (the b21 transcription; registers the kit's levers, applies none)

    install_transition_shard(rows)
    spec = importlib.util.spec_from_file_location("af3jax_rowpair", os.path.join(kit, "inprocess", "rowpair.py"))
    rowpair = importlib.util.module_from_spec(spec)
    sys.modules["af3jax_rowpair"] = rowpair
    spec.loader.exec_module(rowpair)
    # The adapter's NCCL-headroom ESTIMATE (a ceiling of 0.90 at 8 H100s) reads the card's total memory through torch or nvidia-smi, neither
    # present here, so it always lands on its NOTE -- whose blanks then break the kit's own exit line (evidence_error). Off, by name.
    rowpair.MEM_FRACTION_CEILINGS = {}
    print(f"{TAG} n_gpu={n} schedule={schedule} transition_rows={rows} script={os.path.basename(script)} "
          f"headroom_estimate=off(card_total_unreadable_on_rocm) triattn={tile} "
          f"glu={'triton(TOKAMAX_ROCM_TRITON=1)' if os.environ.get('TOKAMAX_ROCM_TRITON') == '1' else 'xla'}", flush=True)
    try:
        rowpair.install(n, b21=big_levers, schedule=schedule)
        mspec = importlib.util.spec_from_file_location(rowpair.SCRIPT_MODULE, script)
        mod = importlib.util.module_from_spec(mspec)
        sys.modules[rowpair.SCRIPT_MODULE] = mod
        sys.argv = [script] + list(flags)
        mspec.loader.exec_module(mod)                                   # defines flags, ModelRunner, main; its __main__ guard stays shut
        if hasattr(mod.ModelRunner, "_jitted_apply"):
            rowpair.patch_runner(mod)                                   # the kit's own runner
        else:
            bind_stock_runner(mod, rowpair._STATE["rmesh"])            # the stock runner (this stack)
        from absl import app
        app.run(mod.main)
    finally:
        if reporter is not None:
            print(f"{TAG} report triattn_flash: {reporter.report()}", flush=True)
        peaks()
        host_peak()


if __name__ == "__main__":
    main()
