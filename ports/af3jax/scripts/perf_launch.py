#!/usr/bin/env python3
"""perf_launch.py SCRIPT [FLAGS...] -- stock run_alphafold.py on ONE GCD with the kit's pure-JAX speed levers (performance plan, item C1).

None of these is a kernel; each rebinds stock AlphaFold 3 functions on the tree this stack runs (the recipe's eleven pinned source hashes
match it byte for byte). PERF_LEVERS picks any subset, comma-separated; they install in the kit's own order (af3_jax_opt/fpf_launch.py:
TREE_LEVERS first, then the add-on's hoist, then INSTALL_AFTER_ADDON):

  cond_share       COND_SHARE (inprocess/cond_share.py): the sampler's per-step conditioning computed once per step and shared by the
                   num_samples samples, instead of once per sample. REPLACES diffusion_head.sample, so it goes first.
  atom_cond_hoist  ATOM_COND_HOIST (inprocess/atom_cond_hoist.py): the atom encoder's step-invariant conditioning once per sample call.
                   Rebinds diffusion_head.DiffusionHead before the hoist derives its head from it; engages only with 'hoist' (without
                   it the lever steps aside by name, aside=no_hoist_step).
  sampler_bf16     SAMPLER_BF16 (inprocess/sampler_bf16.py): the sampler's f32 GEMM islands with bf16 operands (tolerance class).
  hoist            the add-on's diffusion hoist (af3_flashpairformer/diffusion_hoist.py): the pair conditioning and, on the OF3 layout
                   (--of3_weights), the 24 blocks' pair logits once per sample call instead of at every one of the 200 steps. Resident
                   cost ~(24 x 16 x 4 + 128 x 4) x N^2 bytes (f32 logits + pair conditioning: ~4 GiB at N=1,664, ~15 GiB at 3,200).
                   The file is loaded BY PATH under its package name: the package __init__ would import the add-on's NVIDIA-tiled
                   Triton kernels (which need the kit's opt_core) and query the device for a tile table at import. The hoist module
                   itself imports only jax, haiku and alphafold3; HOIST_LOGITS finds it in sys.modules by that name.
  triattn_flash    C2 (this port, not the kit): the triangle-attention core (GridSelfAttention, every stack) through tokamax's Pallas-Triton
                   flash attention with its tile PINNED for gfx90a (PERF_TRIATTN_CFG=block_q:block_k:num_warps:num_stages, default 128:64:4:1,
                   the C2a sweep's best: 5-6x per call at 1,664-3,200 tokens, bf16-class output, less memory). The pin rebinds tokamax's
                   implementation table with PallasTritonFlashAttention(config=...), which tokamax resolves before any heuristic or cache; the
                   launcher appends --flash_attention_implementation=triton (absl keeps the last). In this fork that flag reaches ONE call,
                   modules.GridSelfAttention's tokamax.dot_product_attention (the other attentions are plain einsums), so nothing else
                   changes. Its report counts the trace-time calls that reached the op, with shapes and tiles. The op is told it is
                   supported on ROCm directly: tokamax's own gate, our patch's TOKAMAX_ROCM_TRITON=1, would ALSO move every
                   tokamax.gated_linear_unit (the transitions) from XLA onto a Triton kernel -- leave it unset (the installed line says glu=).
  hoist_logits     HOIST_LOGITS (inprocess/hoist_logits.py): the hoisted logits indexed in place by the 24 blocks (float32 store =
                   value-identical to the hoist alone). PERF_HOIST_LOGITS_DTYPE=bfloat16 stores them in bf16, halving the resident
                   logits (tolerance class). Needs 'hoist'.

PERF_MEM_LEVERS (D1, after M1 found the single-GCD peak in the diffusion head's pair conditioning): a comma list of the kit's own
--mode big memory levers (af3_jax_opt/big_levers.py), installed through the kit's core exactly as big_launch.py does it (opt_core.mem.apply
with the kit's hooks and default settings, in the kit's order, strict: a refusal stops the launch by name) -- none of big mode's NVIDIA
base comes with them. Each is registered by the kit with the reason it keeps values:
  samples_per_pass  the vmapped diffusion sampler run k samples per pass (k=1), stock draws kept
  transition_shard  the pair transitions in row shards (256 rows)
  cond_shard        the diffusion head's pair conditioning (concat, norm, projection, two transitions) in row shards (256 rows)
  logits_shard      the diffusion transformer's per-block pair LayerNorm + logits projection in row shards
  trimul_chunk      TriangleMultiplication (pairformer, MSA stack, confidence pairformer) in row blocks (512 rows)
Needs AF3JAX_CORE (common/opt_core). The record of what applied prints up front, the kit's traced-site counts at exit.

PERF_HOIST_MAX_TOKENS=N (optional) gates the hoist at trace time: a padded bucket above N runs the saved stock sampler (the hoist's head
and transformer default to mode='stock', so that path is stock end to end, and ATOM_COND_HOIST / HOIST_LOGITS step aside with it).
Environment: AF3JAX_KIT (af3_jax/opt/af3_jax_opt), AF3JAX_ADDON (af3_jax/opt/forward/flashpairformer), PERF_LEVERS,
PERF_HOIST_LOGITS_DTYPE, PERF_HOIST_MAX_TOKENS, PERF_ATTN_CHUNK (triangle attention's rows per call: none | R | max:rows,...), PERF_PAIRLOGITS_ROWS (the diffusion transformer's pair-logits projection in R-token-row chunks: G6/G7's hipBLASLt fault), PERF_TRIATTN_CFG and PERF_TRIATTN_IMPL (triattn_flash: triton, the default, or the
xla_chunked control). Prints one
``[perf-launch] installed`` line up front, one ``hoist gate`` line per trace when gated, the levers' own report() at exit, and
per-device allocator peaks from a ``finally`` block.
"""
import importlib.util
import os
import runpy
import sys
import time

T0 = time.time()
TAG = "[perf-launch]"
KNOWN = ("cond_share", "atom_cond_hoist", "sampler_bf16", "hoist", "hoist_logits", "triattn_flash")
MEM_KNOWN = ("samples_per_pass", "transition_shard", "cond_shard", "logits_shard", "trimul_chunk")   # the kit's big line, its order
PASS1 = ("cond_share", "atom_cond_hoist", "sampler_bf16")          # before the hoist (kit order)
HOIST_MODULE = "af3_flashpairformer.diffusion_hoist"               # the name HOIST_LOGITS looks up in sys.modules


def peaks() -> None:
    try:
        import jax
        for d in jax.local_devices():
            st = d.memory_stats() or {}
            pk, lim = st.get("peak_bytes_in_use"), st.get("bytes_limit")
            gib = lambda b: (b / 2**30) if isinstance(b, int) else float("nan")  # noqa: E731
            print(f"[peak] device={d.id} peak_bytes_in_use={pk} bytes_limit={lim} peak_GiB={gib(pk):.2f} limit_GiB={gib(lim):.2f} "
                  f"wall_s={time.time() - T0:.0f}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[peak] unavailable: {type(e).__name__}: {e}", flush=True)


def load_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                                          # registered before exec (the kit's modules resolve through it)
    spec.loader.exec_module(mod)
    return mod


def gate_hoist(dh, max_tokens: int) -> None:
    """Above max_tokens (the padded bucket, read at trace time) run the stock sampler saved by the hoist's install()."""
    import haiku as hk
    from alphafold3.model import model as M
    hoisted, stock = M.Model._sample_diffusion, dh._STOCK["sample"]

    def _sample_diffusion(self, batch, embeddings, *args, **kwargs):
        n = int(embeddings["single"].shape[-2])
        use = n <= max_tokens
        print(f"{TAG} hoist gate tokens={n} max={max_tokens} -> {'hoisted' if use else 'stock'}", flush=True)
        return (hoisted if use else stock)(self, batch, embeddings, *args, **kwargs)

    M.Model._sample_diffusion = hk.transparent(_sample_diffusion)


def pin_triattn(cfg: str, impl: str = "triton"):
    """tokamax resolves an op's config from the op's own `config` field first: rebind its 'triton' attention with the gfx90a tile.
    impl='xla_chunked' puts tokamax's kernel-free chunked XLA attention in that slot instead (a control: another non-bitwise but
    closer-to-stock triangle attention, C2a relative RMS 2.7e-3 against 9.6e-3 for the flash kernel).
    The op's _fwd is wrapped to count the trace-time calls that reach it, with the query shape and the config each one received, so the
    report shows the pin engaged at every call site (template, MSA and main stacks, confidence head) rather than assuming it."""
    import collections
    import functools
    import types
    import immutabledict
    from tokamax._src.ops.attention import api, pallas_triton as pt
    impls = dict(api.IMPLEMENTATIONS)
    if impl == "triton":
        bq, bk, w, st = (int(x) for x in cfg.split(":"))
        op, label = pt.PallasTritonFlashAttention(config=pt.Config(block_q=bq, block_k=bk, num_warps=w, num_stages=st)), f"{bq}:{bk}:{w}:{st}"
        stock_supported = pt.PallasTritonFlashAttention.supported_on

        def supported_on(self, device):    # tokamax asks has_triton_support(), which our patch keeps False on ROCm unless
            if str(getattr(device, "compute_capability", "")).startswith("gfx"):   # TOKAMAX_ROCM_TRITON=1 -- a switch that also moves
                return True                                                        # every gated linear unit onto Triton: not wanted
            return stock_supported(self, device)

        pt.PallasTritonFlashAttention.supported_on = supported_on
    elif impl == "xla_chunked":
        op = impls["xla_chunked"]
        label = f"xla_chunked(chunk_size={op.chunk_size})"
    else:
        sys.exit(f"{TAG} PERF_TRIATTN_IMPL={impl!r}: triton or xla_chunked")
    impls["triton"] = op
    api.IMPLEMENTATIONS = immutabledict.immutabledict(impls)
    cls = type(op)
    seen, fwd = collections.Counter(), cls._fwd
    tile_of = lambda c: (":".join(str(getattr(c, a)) for a in ("block_q", "block_k", "num_warps", "num_stages"))  # noqa: E731
                         if all(hasattr(c, a) for a in ("block_q", "block_k", "num_warps", "num_stages")) else type(c).__name__)

    @functools.wraps(fwd)                  # tokamax binds call arguments against inspect.signature(op._fwd): keep the original's
    def _fwd(self, *args, **kwargs):
        q = args[0] if args else kwargs.get("q")
        seen[(tuple(getattr(q, "shape", ())), str(getattr(q, "dtype", "?")), tile_of(kwargs.get("config")))] += 1
        return fwd(self, *args, **kwargs)

    cls._fwd = _fwd
    report = lambda: f"{cls.__name__} traced {sum(seen.values())} calls: " + ", ".join(  # noqa: E731
        f"q{list(s)} {d} cfg={t} x{n}" for (s, d, t), n in sorted(seen.items()))
    return types.SimpleNamespace(report=report), label


def install_mem_levers(names, kit: str, core: str):
    """The kit's --mode big memory levers, the way big_launch.py applies them (opt_core.mem.apply, strict), minus its NVIDIA base."""
    for p in (core, kit):
        if p not in sys.path:
            sys.path.insert(0, p)
    import big_levers
    from opt_core import mem
    from opt_core.mem import registry
    order = [n for n in big_levers.LINE if n in names]
    ctx = registry.Ctx(prefix="AF3_JAX", tag="perf-launch", framework="jax", hooks=big_levers.HOOKS,
                       settings={k: dict(v) for k, v in big_levers.SETTINGS.items()}, environ=os.environ, graphs=False)
    rec = mem.apply(order, ctx, base="none", strict=True)
    words = ", ".join(f"{a.lever}{a.settings}" for a in rec.applied)
    return big_levers, words


def pin_attn_chunk(spec: str):
    """PERF_ATTN_CHUNK: triangle attention's rows per call. GridSelfAttention calls its attention in chunks of
    get_shard_size(N, global_config.pair_attention_chunk_size) rows (the fork: 128 up to 1,536 tokens, 32 above), sized for XLA's
    materialised logits; flash never materialises them (P0c, job 50309562: one call over all rows ran the attention 1.21x faster at
    1,664). spec: 'none' = one call over every row; an integer R = R rows per call at every size; or the fork's own form
    'max:rows,...' ('none' allowed in either place). Only the triangle-attention spec is replaced (the pair-transition shard spec is a
    different tuple); each row's attention is the same arithmetic, so values should not change, only call sizes and the projections'
    memory. Returns (SimpleNamespace(report=...), the spec installed)."""
    import collections
    import types
    from alphafold3.model import model_config
    from alphafold3.model.network import modules

    def tok(x):
        return None if x.strip().lower() == "none" else int(x)
    if ":" in spec:
        chunk = tuple((tok(a), tok(b)) for a, b in (part.split(":") for part in spec.split(",")))
    else:
        chunk = ((None, tok(spec)),)
    default = tuple(tuple(p) for p in model_config.GlobalConfig().pair_attention_chunk_size)
    orig, seen = modules.get_shard_size, collections.Counter()

    def get_shard_size(num_residues, shard_spec):
        if tuple(tuple(p) for p in shard_spec) == default:
            out = orig(num_residues, chunk)
            seen[(int(num_residues), out)] += 1
            return out
        return orig(num_residues, shard_spec)

    modules.get_shard_size = get_shard_size
    report = lambda: f"default {default} -> {chunk}; traced calls (N, rows per call): " + (  # noqa: E731
        ", ".join(f"{n}:{r} x{c}" for (n, r), c in sorted(seen.items(), key=str)) or "none")
    return types.SimpleNamespace(report=report), chunk


def pin_pairlogits_rows(rows: int):
    """PERF_PAIRLOGITS_ROWS: the diffusion transformer's pair_logits_projection over ROWS token rows at a time. G6 / G7 (jobs 50437098,
    50456060): hipBLASLt's default algorithm computes that f32 GEMM ([N^2, 128] x [128, 64], column-major result: the [2, 3, 0, 1]
    transpose below) wrongly from 18,612,224 rows, i.e. from about 4,314 tokens (first bad row 2^29 - 28 x rows), and correctly at and
    below 18,546,688; every chunk of ROWS x N rows stays far below that. A subclass of the fork's Transformer (Haiku wraps methods at class
    creation, so a rebound method would run outside the module's scope; the kit's LOGITS_SHARD takes the same route) whose official-
    parameters branch applies ONE Linear module (the stock name and parameters) to each row chunk of the shared pair_act and writes each
    chunk's logits, transposed to [4, 16, r, N], into a preallocated [4, 16, N, N] buffer. Values per row are the same arithmetic: only
    call sizes change, plus one chunk's output of memory. The OF3-weights branch is verbatim. Refused if another lever already replaced
    Transformer. Returns (SimpleNamespace(report=...), installed)."""
    import collections
    import inspect
    import types
    import jax
    from alphafold3.model.network import diffusion_transformer as DT

    parent = DT.Transformer
    if not inspect.unwrap(parent.__call__).__code__.co_filename.endswith("diffusion_transformer.py"):
        return types.SimpleNamespace(report=lambda: "NOT installed: Transformer was already replaced by another lever"), False
    hk, jnp, hm = DT.hk, DT.jnp, DT.hm
    seen = collections.Counter()

    def __call__(self, act, mask, single_cond, pair_cond):
        assert self.config.num_blocks % self.config.super_block_size == 0
        num_super_blocks = self.config.num_blocks // self.config.super_block_size

        if self.global_config.of3_weights and pair_cond is not None:       # verbatim (diffusion_transformer.py:219-248)
            def block(act):  # pylint: disable=function-redefined
                pair_act = hm.LayerNorm(name="pair_input_layer_norm", use_fast_variance=False, create_offset=False)(pair_cond)
                block_pair_logits = hm.Linear(self.config.attention.num_head, name="pair_logits_projection")(pair_act)
                block_pair_logits = jnp.transpose(block_pair_logits, [2, 0, 1])
                act += DT.self_attention(act, mask, block_pair_logits, self.config.attention, self.global_config, single_cond,
                                         name=self.name)
                act += DT.transition_block(act, self.config.num_intermediate_factor, self.global_config, single_cond, name=self.name)
                return act

            def super_block(act):  # pylint: disable=function-redefined
                return hk.experimental.layer_stack(self.config.super_block_size)(block)(act)

            return hk.experimental.layer_stack(num_super_blocks)(super_block)(act)

        def block(act, pair_logits):                                       # verbatim (:251-268)
            act += DT.self_attention(act, mask, pair_logits, self.config.attention, self.global_config, single_cond, name=self.name)
            act += DT.transition_block(act, self.config.num_intermediate_factor, self.global_config, single_cond, name=self.name)
            return act, None

        if pair_cond is None:
            pair_act = None
        else:
            pair_act = hm.LayerNorm(name="pair_input_layer_norm", use_fast_variance=False, create_offset=False)(pair_cond)

        def super_block(act):                                              # :280-291, the projection in row chunks
            if pair_act is None:
                pair_logits = None
            else:
                sbs, nh = self.config.super_block_size, self.config.attention.num_head
                lin = hm.Linear((sbs, nh), name="pair_logits_projection")
                n = int(pair_act.shape[0])
                if n <= rows:
                    pair_logits = jnp.transpose(lin(pair_act), [2, 3, 0, 1])
                    seen[(n, n)] += 1
                else:
                    pair_logits = None
                    for i in range(0, n, rows):
                        blk = jnp.transpose(lin(pair_act[i:i + rows]), [2, 3, 0, 1])     # [sbs, nh, r, N]
                        if pair_logits is None:
                            pair_logits = jnp.zeros((sbs, nh, n) + tuple(blk.shape[3:]), blk.dtype)
                        pair_logits = jax.lax.dynamic_update_slice(pair_logits, blk, (0, 0, i) + (0,) * (blk.ndim - 3))
                    seen[(n, rows)] += 1
            return hk.experimental.layer_stack(self.config.super_block_size, with_per_layer_inputs=True)(block)(act, pair_logits)

        return hk.experimental.layer_stack(num_super_blocks, with_per_layer_inputs=True)(super_block)(act)[0]

    DT.Transformer = type("Transformer", (parent,), {"__call__": __call__, "__doc__": parent.__doc__, "__module__": parent.__module__})
    report = lambda: f"rows {rows}; traced (N, rows per call): " + (  # noqa: E731
        ", ".join(f"{n}:{r} x{c}" for (n, r), c in sorted(seen.items())) or "none")
    return types.SimpleNamespace(report=report), True


def main() -> None:
    if len(sys.argv) < 2 or not sys.argv[1].endswith(".py"):
        sys.exit("usage: perf_launch.py SCRIPT.py [flags...]")
    script, flags = os.path.abspath(sys.argv[1]), sys.argv[2:]
    kit, addon = os.environ["AF3JAX_KIT"], os.environ["AF3JAX_ADDON"]
    want = [x.strip() for x in os.environ.get("PERF_LEVERS", "").split(",") if x.strip()]
    bad = [x for x in want if x not in KNOWN]
    if bad:
        sys.exit(f"{TAG} unknown lever(s) {bad}; known: {','.join(KNOWN)}")
    if "hoist_logits" in want and "hoist" not in want:
        sys.exit(f"{TAG} hoist_logits needs hoist")
    dtype = os.environ.get("PERF_HOIST_LOGITS_DTYPE", "")
    if dtype not in ("", "float32", "bfloat16"):
        sys.exit(f"{TAG} PERF_HOIST_LOGITS_DTYPE={dtype!r}: float32 or bfloat16")
    gate = int(os.environ["PERF_HOIST_MAX_TOKENS"]) if os.environ.get("PERF_HOIST_MAX_TOKENS") else None
    mem_want = [x.strip() for x in os.environ.get("PERF_MEM_LEVERS", "").split(",") if x.strip()]
    bad = [x for x in mem_want if x not in MEM_KNOWN]
    if bad:
        sys.exit(f"{TAG} unknown memory lever(s) {bad}; known: {','.join(MEM_KNOWN)}")
    if "cond_share" in want and "samples_per_pass" in mem_want:     # the kit's chunked sampler re-implements the loop with the fork's own
        sys.exit(f"{TAG} cond_share with samples_per_pass: the chunked sampler (k=1) re-implements the sampling loop and never calls "
                 "cond_share's sampler, so cond_share would be a silent no-op -- drop one of them")
    os.environ["AF3_FLASHPAIRFORMER"] = "off"                       # guard: were the add-on package imported anyway, its kernels stay off
    sys.path.insert(0, os.path.dirname(script))                      # as `python SCRIPT` would

    mods, installed, tile = {}, {}, ""
    for name in PASS1:                                               # pass 1, kit order
        if name in want:
            mod = load_path(f"af3jax_perf_{name}", os.path.join(kit, "inprocess", f"{name}.py"))
            os.environ[mod.ENV_SWITCH] = "1"
            installed[name] = bool(mod.install())
            mods[name] = mod
    if "hoist" in want:                                              # the add-on's hoist (what its __init__'s install_hoist() calls)
        dh = load_path(HOIST_MODULE, os.path.join(addon, "af3_flashpairformer", "diffusion_hoist.py"))
        installed["hoist"] = dh.install() == "hoist" and bool(dh._STOCK)
        if gate is not None:
            gate_hoist(dh, gate)
    if "hoist_logits" in want:                                       # pass 2: subclasses the hoist's HoistTransformer
        mod = load_path("af3jax_perf_hoist_logits", os.path.join(kit, "inprocess", "hoist_logits.py"))
        if dtype:
            mod.STORE_DTYPE = dtype
        os.environ[mod.ENV_SWITCH] = "1"
        installed["hoist_logits"] = bool(mod.install())
        mods["hoist_logits"] = mod
    big, mem_words = None, "-"
    if mem_want:                                                     # D1: the kit's memory levers, after the diffusion levers (big_launch's order)
        big, mem_words = install_mem_levers(mem_want, kit, os.environ["AF3JAX_CORE"])
    if "triattn_flash" in want:                                      # C2: pinned flash attention for the triangle-attention core
        mods["triattn_flash"], tile = pin_triattn(os.environ.get("PERF_TRIATTN_CFG", "128:64:4:1"),
                                                  os.environ.get("PERF_TRIATTN_IMPL", "triton"))
        installed["triattn_flash"] = bool(tile)
        flags = list(flags) + ["--flash_attention_implementation=triton"]
    chunk_words = "-"
    if os.environ.get("PERF_ATTN_CHUNK"):                            # P0c: triangle attention's rows per call
        mods["attn_chunk"], chunk_spec = pin_attn_chunk(os.environ["PERF_ATTN_CHUNK"])
        chunk_words = str(chunk_spec).replace(" ", "")
    pl_words = "-"
    pl_rows = int(os.environ.get("PERF_PAIRLOGITS_ROWS") or 0)       # G6/G7/F1: the diffusion pair-logits projection in row chunks
    if pl_rows > 0:                                                  # (0 or unset: off)
        mods["pairlogits_rows"], ok = pin_pairlogits_rows(pl_rows)
        pl_words = str(pl_rows) if ok else "REFUSED"
    words = " ".join(f"{k}={'on' if v else 'NOT_INSTALLED'}" for k, v in installed.items()) or "none(stock)"
    print(f"{TAG} installed {words} logits_dtype={(dtype or 'float32') if 'hoist_logits' in want else '-'} "
          f"hoist_max_tokens={gate if 'hoist' in want and gate is not None else '-'} "
          f"triattn_cfg={tile if 'triattn_flash' in want else '-'} "
          f"glu={'triton(TOKAMAX_ROCM_TRITON=1)' if os.environ.get('TOKAMAX_ROCM_TRITON') == '1' else 'xla'} "
          f"mem_levers={mem_words} attn_chunk={chunk_words} pairlogits_rows={pl_words} "
          f"script={os.path.basename(script)}", flush=True)

    sys.argv = [script] + list(flags)
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        for name, mod in mods.items():
            try:
                print(f"{TAG} report {name}: {mod.report()}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"{TAG} report {name}: unavailable ({type(e).__name__})", flush=True)
        if big is not None:
            try:
                print(f"{TAG} report mem_levers: traced={dict(getattr(big, 'TRACED', {}))} notes={big.exit_notes()}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"{TAG} report mem_levers: unavailable ({type(e).__name__}: {e})", flush=True)
        peaks()


if __name__ == "__main__":
    main()
