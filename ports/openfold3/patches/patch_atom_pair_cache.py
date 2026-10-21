#!/usr/bin/env python3
"""Cache the atom-level pair path across the denoising steps of a roll-out (env-gated; off by default).

WHY
  Inside one roll-out the atom encoder's pair representation reads nothing that changes between denoising steps. In
  AtomAttentionEncoder.get_atom_reps, plm is built from:
    - the reference-conformer embedding,
    - the trunk single term,
    - the roll-out's plm_z (already cached by the tp line),
    - the single-conditioning pair term,
    - the pair MLP.
  Only ql reads the noisy coordinates rl. Neither do:
    - the two atom transformers' pair LayerNorm (DiffusionTransformer.forward: layer_norm_z, once per call), nor
    - each block's pair-bias projection over it (CrossAttentionPairBias._prep_block_inputs: linear_z + permute).
  OpenFold3 recomputes all of it on every step, for the encoder and the decoder. At 2,310 res / P=4 the atom-pair LayerNorm
  (normalised width 16 over [1, 1, 556, 32, 128, 16], a slow shape for PyTorch's kernel) runs twice a step at ~10 ms each, and
  with the projections and adds the path is ~25-30 ms of a ~120 ms step (job 50324562's kernel table grouped by input shape).

WHAT
  ROWPAIR_ATOM_PAIR_CACHE=1 adds, inside the tp line's sharded_denoiser (one roll-out), three instance swaps:
    get_atom_reps        on the roll-out's first step the stock method runs and its (cl, plm) are kept. Later steps return
                         (cl + linear_r(rl), cl, plm): the noisy-position embedder's ql statement on the same cl.
    atom_transformer     (encoder's and decoder's) the stock forward with the pair LayerNorm of the kept plm computed once.
                         The blocks run in turn exactly as checkpoint_blocks runs them without grad.
    _prep_block_inputs   (every block) the stock statements, with the linear_z + permute of the kept normalised pair computed
                         once. The query / key windows and the mask bias are computed every step.
  Nothing is computed differently, so outputs are bitwise those without it.
  Cost: the kept tensors stay resident for the roll-out: plm, two normalised pairs, six per-block biases. That is ~0.7 GB at
  2,310 res (replicated on every rank), growing with the atom count.
  Unset = sharded_denoiser's four swaps only, as before. Inference only: refused under autograd.

USAGE  python3 patch_atom_pair_cache.py --apply | --check | --revert [path]
"""
import os, sys, pathlib

P = pathlib.Path('/scratch/pawsey1018/gbouras/af3-setonix/of3_opt_src/opt/openfold3_opt/tp_rowpair/diffusion.py')
if len(sys.argv) > 2:
    P = pathlib.Path(sys.argv[2])
MARK = '# [atom_pair_cache]'
BACKUP = P.with_suffix('.py.pre_atom_pair_cache')

ANCHOR = "@contextmanager\ndef sharded_denoiser(dm, cache: RolloutCache):\n"
HELPERS = '''# [atom_pair_cache] ROWPAIR_ATOM_PAIR_CACHE=1: inside one roll-out the atom encoder's pair path (reference-conformer embedding, trunk single
# term, plm_z, the single-conditioning pair term, the pair MLP: AtomAttentionEncoder.get_atom_reps; only ql reads rl), the two atom
# transformers' pair LayerNorm and every block's pair-bias projection over it read nothing that changes between denoising steps. With the
# gate on they are computed on the roll-out's first step and reused; ql, the blocks' query / key windows and the mask bias are computed
# every step by the same statements. Bitwise unchanged. Unset = the four swaps of sharded_denoiser only, as before.
_ATOM_PAIR_CACHE = os.environ.get("ROWPAIR_ATOM_PAIR_CACHE", "") == "1"


@contextmanager
def swapped_attr(obj, name, fn):
    """``obj.<name>`` = ``fn`` on this INSTANCE for the duration (the class and every other instance untouched). [atom_pair_cache]"""
    had = name in obj.__dict__
    prev = obj.__dict__.get(name)
    setattr(obj, name, fn)
    try:
        yield obj
    finally:
        if had:
            setattr(obj, name, prev)
        else:
            delattr(obj, name)


def atom_pair_cache_swaps(dm, stack):
    """[atom_pair_cache] Enter one roll-out's atom-pair swaps on ``stack`` (an ExitStack scoped to sharded_denoiser); returns the store."""
    from openfold3.core.utils.atom_attention_block_utils import convert_single_rep_to_blocks
    if torch.is_grad_enabled():
        raise DiffusionRefused("refused: ROWPAIR_ATOM_PAIR_CACHE under autograd (the roll-out is inference only)")
    enc, dec = dm.atom_attn_enc, dm.atom_attn_dec
    npe = enc.noisy_position_embedder
    store = {"reps": None, "ln": {}, "bias": {}, "hits": 0}
    stock_reps = enc.get_atom_reps                            # the bound stock method; its noisy_position_embedder is the roll-out's swap

    def get_atom_reps(batch, rl=None, si_trunk=None, zij_trunk=None):
        if rl is None:
            return stock_reps(batch, rl, si_trunk, zij_trunk)
        if store["reps"] is None:
            ql, cl, plm = stock_reps(batch, rl, si_trunk, zij_trunk)
            store["reps"] = (cl, plm)
            return ql, cl, plm
        cl, plm = store["reps"]
        store["hits"] += 1
        return cl + npe.linear_r(rl), cl, plm                 # noisy_position_embedder_forward_band's ql statement, on the same cl

    def transformer_forward(tr):
        def forward(a, s, z, mask=None, use_deepspeed_evo_attention=False, use_cueq_triangle_kernels=False, use_triton_triangle_kernels=False,
                    use_lma=False, use_high_precision_attention=False, _mask_trans=True):
            reps = store["reps"]
            if reps is not None and z is reps[1]:
                zn = store["ln"].get(id(tr))
                if zn is None:
                    zn = store["ln"][id(tr)] = tr.layer_norm_z(z)
            else:
                zn = tr.layer_norm_z(z)
            for b in tr.blocks:                                # checkpoint_blocks without grad: each block in turn on (a,)
                a = b(a, s=s, z=zn, mask=mask, use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                      use_cueq_triangle_kernels=use_cueq_triangle_kernels, use_triton_triangle_kernels=use_triton_triangle_kernels,
                      use_lma=use_lma, use_high_precision_attention=use_high_precision_attention, _mask_trans=_mask_trans)
            return a
        return forward

    def prep_block_inputs(apb):
        stock_prep = apb._prep_block_inputs
        def prep(a, z, mask):
            ent = store["bias"].get(id(apb))
            if ent is not None and ent[0] is z:
                a_query, a_key, m = convert_single_rep_to_blocks(ql=a, n_query=apb.n_query, n_key=apb.n_key, atom_mask=mask)
                return a_query, a_key, [(apb.inf * (m - 1))[..., None, :, :], ent[1]]
            a_query, a_key, biases = stock_prep(a=a, z=z, mask=mask)
            if any(z is v for v in store["ln"].values()):
                store["bias"][id(apb)] = (z, biases[1])
            return a_query, a_key, biases
        return prep

    stack.enter_context(swapped_attr(enc, "get_atom_reps", get_atom_reps))
    for tr in (enc.atom_transformer, dec.atom_transformer):
        stack.enter_context(swapped_attr(tr, "forward", transformer_forward(tr)))
        for b in tr.blocks:
            apb = getattr(b, "attention_pair_bias", None)
            if apb is not None and hasattr(apb, "_prep_block_inputs"):
                stack.enter_context(swapped_attr(apb, "_prep_block_inputs", prep_block_inputs(apb)))
    return store


'''
OLD_WITH = ("    with swapped_forward(dm, dm_forward), swapped_forward(dc, dc_forward), swapped_forward(dt, dt_forward), swapped_forward(npe, npe_forward):\n"
            "        yield dm\n")
NEW_WITH = ("    from contextlib import ExitStack                          # [atom_pair_cache]\n"
            "    with swapped_forward(dm, dm_forward), swapped_forward(dc, dc_forward), swapped_forward(dt, dt_forward), swapped_forward(npe, npe_forward), \\\n"
            "            ExitStack() as _aps:\n"
            "        if _ATOM_PAIR_CACHE:\n"
            "            atom_pair_cache_swaps(dm, _aps)\n"
            "        yield dm\n")


def _apply(s):
    counts = {"anchor": s.count(ANCHOR), "with": s.count(OLD_WITH)}
    if any(n != 1 for n in counts.values()):
        raise SystemExit(f'ANCHORS NOT FOUND EXACTLY ONCE ({counts}) -- not patching')
    for need in ('\nimport os\n', '\nimport torch\n', 'class DiffusionRefused('):
        if need not in s:
            raise SystemExit(f'diffusion.py lacks {need.strip()!r} -- not patching')
    new = s.replace(ANCHOR, HELPERS + ANCHOR, 1).replace(OLD_WITH, NEW_WITH, 1)
    compile(new, str(P), 'exec')
    return new


def _write(text):
    tmp = P.with_suffix('.py.atom_pair_cache_tmp')
    tmp.write_text(text)
    os.replace(tmp, P)                                     # atomic: importers never see a half-written file


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else '--check'
    s = P.read_text()
    applied = MARK in s
    if mode == '--check':
        print(f'atom_pair_cache: applied={applied}  backup={BACKUP.exists()}')
        return 0
    if mode == '--apply':
        if applied:
            print('already applied'); return 0
        BACKUP.write_text(s)
        _write(_apply(s))
        print(f'applied (backup: {BACKUP.name})'); return 0
    if mode == '--revert':
        if not BACKUP.exists():
            print('no backup', file=sys.stderr); return 1
        _write(BACKUP.read_text())
        print('reverted'); return 0
    print('usage: --apply | --check | --revert [path]', file=sys.stderr); return 2


sys.exit(main())
