#!/usr/bin/env python3
# Port the tp_rowpair adapter from OpenFold3 0.4.1 to 0.5.0.
#
#     python port/port_of3_050.py --apply  <path-to-openfold3_opt/tp_rowpair>
#     python port/port_of3_050.py --check  <path>
#     python port/port_of3_050.py --revert <path>
#
# Idempotent. Keeps .orig backups. Syntax-checks after applying.
#
# ---------------------------------------------------------------------------
# FIX 1 -- AdaLN detection (4 sites: pairstack x2, diffusion x2)
# ---------------------------------------------------------------------------
# 0.4.1 had ONE AttentionPairBias with a `use_ada_layer_norm: bool`. 0.5.0 split
# it into AttentionPairBias (plain LayerNorm) and DiffusionAttentionPairBias
# (AdaLN + a linear_ada_out gate) and dropped the flag, so reading it raises:
#     AttributeError: 'AttentionPairBias' object has no attribute 'use_ada_layer_norm'
# Both releases expose `linear_ada_out` exactly when AdaLN is in use, so that is
# the portable test.
#
# ---------------------------------------------------------------------------
# FIX 2 -- ending-node triangle-bias orientation (the numerical bug)
# ---------------------------------------------------------------------------
# 0.5.0 added `transpose_bias` to TriangleAttention.forward, and PairBlock calls
# the ENDING node with it:
#     z = z.transpose(-2, -3)                 # 0.4.1 also did .contiguous()
#     tri_att_end(z, mask=pair_mask.transpose(-1,-2), transpose_bias=True, ...)
# which swaps the last two dims of the bias permutation:
#     0.4.1  permute(linear_z(x), (2,0,1)) -> bias[h,j,k] = X[j,k,h]
#     0.5.0  permute(linear_z(x), (2,1,0)) -> bias[h,j,k] = X[k,j,h]
# Upstream's words: the bias should "retain the ordering of the original
# un-transposed input tensor".
#
# The adapter reimplements this statement, so on 0.5.0 it fed the kernel 0.4.1's
# orientation -- finite output, wrong values. That is the observed
# `pairblock maxabs 0.0195` against TOL 1e-5.
#
# opt_core already anticipated the variant: TriAttFns(..., bias_transposed=),
# documented as "an engine that projects the ending node's bias in z's own
# frame, transpose_bias". The adapter hardcoded False for both nodes.
#
# TWO paths need it:
#   * JIT bias schedule -- the core reads fns.bias_transposed itself
#   * GATHER schedule   -- the adapter's own `attend` builds the operand, so the
#                          swap must happen there too
# Detection is by forward-signature, so 0.4.1 keeps its old behaviour.
#
# ---------------------------------------------------------------------------
# FIX 3 -- DiffusionTransformer.use_cross_attention moved to the blocks
# ---------------------------------------------------------------------------
# 0.4.1:  DiffusionTransformer.__init__ set `self.use_cross_attention = n_query
#         is not None`, and gated `z = self.layer_norm_z(z)` on it.
# 0.5.0:  the attribute is gone from the parent (it survives on each
#         DiffusionTransformerBlock) and layer_norm_z is applied unconditionally.
#
# The adapter reads it as a REFUSAL guard:
#     if dt.use_cross_attention: raise DiffusionRefused(...)
# so on 0.5.0 it raises AttributeError instead.
#
# NOTE: the obvious `getattr(dt, "use_cross_attention", True)` is WRONG here --
# defaulting to True would refuse every diffusion run. The flag must be read from
# the blocks, which still carry `n_query is not None`.
#
# ---------------------------------------------------------------------------
# FIX 4 -- _supports_transpose_bias must tolerate a module without .forward
# ---------------------------------------------------------------------------
# The census tests inject recording fakes (types.SimpleNamespace) as core seams.
# `inspect.signature(ta.forward)` raises AttributeError on those, which FIX 2's
# helper did not catch -- a regression this patch introduced.
#
# ---------------------------------------------------------------------------
# FIX 6 -- sample_rotations moved out of diffusion_module
# ---------------------------------------------------------------------------
# 0.4.1: openfold3.core.model.structure.diffusion_module.sample_rotations
# 0.5.0: openfold3.core.model.structure.augmentation.sample_rotations
# Same signature (shape, dtype, device); only the module changed. The adapter
# calls it as `_dm.sample_rotations(...)` in the diffusion roll-out, so on 0.5.0:
#     AttributeError: module '...diffusion_module' has no attribute 'sample_rotations'
# This is the FIRST failure past the sharded trunk -- all recycles, the template
# stack, the MSA module and the confidence park completed before it.
#
# ---------------------------------------------------------------------------
# FIX 5 -- the DiT pair-bias LayerNorm moved from the blocks to the transformer
# ---------------------------------------------------------------------------
# 0.4.1: every AttentionPairBias carried `layer_norm_z` and _prep_bias did
#        `linear_z(layer_norm_z(z))` -- once PER BLOCK. DiffusionTransformer's own
#        `layer_norm_z` was gated on use_cross_attention, so the token-level DiT
#        never applied it.
# 0.5.0: DiffusionAttentionPairBias has NO `layer_norm_z` (_prep_bias is just
#        `linear_z(z)`), and DiffusionTransformer.layer_norm_z is applied
#        UNCONDITIONALLY, once, before the block loop.
#
# Net: the norm moved from N blocks to one place. The adapter reads
# `apb.layer_norm_z` in two spots -- the `bias` closure and the fused DitBias
# LN+projection kernel -- so on 0.5.0 it raises:
#     AttributeError: 'DiffusionAttentionPairBias' object has no attribute 'layer_norm_z'
#
# The fix is three-part:
#   * `bias` applies the LN only when the block has one
#   * the fused `bias_into` path is SKIPPED when it does not. Passing
#     ln_weight=None would not disable the kernel's LayerNorm -- per the core's
#     docstring that means an UNWEIGHTED LN, which 0.5.0 does not want, since the
#     norm already happened upstream. Falling back to the engine statement is the
#     correct conservative choice (and on ROCm these kernels fall back anyway).
#   * `diffusion_transformer_rows` applies dt.layer_norm_z to the local pair rows
#     once, matching stock. LayerNorm is per-element over channels, so it shards
#     with no communication.

from __future__ import annotations

import argparse
import ast
import pathlib
import shutil
import sys

HELPER_ADA = '''

def _uses_ada_ln(apb) -> bool:
    """Does this AttentionPairBias apply adaptive LayerNorm?

    OF3 0.4.1: one class with a ``use_ada_layer_norm`` flag.
    OF3 0.5.0: the flag is gone -- ``AttentionPairBias`` is plain LayerNorm and
    ``DiffusionAttentionPairBias`` is AdaLN. Both expose ``linear_ada_out``
    exactly when AdaLN is in use. (af3-setonix port_of3_050.py)
    """
    flag = getattr(apb, "use_ada_layer_norm", None)
    if flag is not None:
        return bool(flag)
    return hasattr(apb, "linear_ada_out")
'''

HELPER_TB = '''

def _supports_transpose_bias(ta) -> bool:
    """Does this TriangleAttention take ``transpose_bias``? (OF3 0.5.0+, not 0.4.1)

    0.5.0's PairBlock calls the ending node with ``transpose_bias=True``, which
    permutes the triangle bias (2,1,0) instead of (2,0,1). The driver must match
    it or the ending-node attention reads a transposed bias plane.
    (af3-setonix port_of3_050.py)
    """
    fwd = getattr(ta, "forward", None)          # census fakes have no .forward
    if fwd is None:
        return False
    try:
        import inspect
        return "transpose_bias" in inspect.signature(fwd).parameters
    except (TypeError, ValueError):
        return False
'''


HELPER_CROSS = '''

def _dt_uses_cross_attention(dt) -> bool:
    """Does this DiffusionTransformer use cross attention?

    OF3 0.4.1 set ``use_cross_attention`` on the DiffusionTransformer itself.
    0.5.0 removed it from the parent -- it survives on each
    ``DiffusionTransformerBlock`` (``n_query is not None``). Defaulting to True
    when absent would refuse every diffusion run, so read it from the blocks.
    (af3-setonix port_of3_050.py)
    """
    flag = getattr(dt, "use_cross_attention", None)
    if flag is not None:
        return bool(flag)
    blocks = getattr(dt, "blocks", None) or ()
    if len(blocks):
        return bool(getattr(blocks[0], "use_cross_attention", False))
    return False
'''


HELPER_DITLN = '''

def _apb_ln_z(apb, z_rows):
    """``layer_norm_z(z_rows)`` when the block carries that LayerNorm, else z_rows.

    OF3 0.4.1 normalised the pair bias in every AttentionPairBias; 0.5.0 removed
    it from DiffusionAttentionPairBias and normalises once in
    DiffusionTransformer.forward instead. (af3-setonix port_of3_050.py)
    """
    ln = getattr(apb, "layer_norm_z", None)
    return z_rows if ln is None else ln(z_rows)


def _dt_norms_z_once(dt) -> bool:
    """Does this DiffusionTransformer own the pair-bias LayerNorm (OF3 0.5.0+)?

    True when the blocks' attention_pair_bias has no ``layer_norm_z`` but the
    transformer does -- 0.5.0's arrangement, where the norm is applied once
    before the block loop. (af3-setonix port_of3_050.py)
    """
    blocks = getattr(dt, "blocks", None) or ()
    if not len(blocks):
        return False
    apb = getattr(blocks[0], "attention_pair_bias", None)
    return (apb is not None and getattr(apb, "layer_norm_z", None) is None
            and getattr(dt, "layer_norm_z", None) is not None)
'''

TB_OLD = ("        tb = tb_full.movedim(-1, 0).unsqueeze(0)                                       "
          "# [1, H, N, N]      (== triangle_bias.unsqueeze(-4) on the row batch)")
TB_NEW = (TB_OLD + "\n"
          "        if _bias_t:                                                                    "
          "# OF3 0.5.0 ending node: bias[h,j,k] = plane[k,j,h] (transpose_bias)\n"
          "            tb = tb.transpose(-1, -2)")

EDITS = {
    "pairstack.py": [
        # FIX 1
        ("    a_ln = apb.layer_norm_a(a, s) if apb.use_ada_layer_norm else apb.layer_norm_a(a)",
         "    a_ln = apb.layer_norm_a(a, s) if _uses_ada_ln(apb) else apb.layer_norm_a(a)"),
        ("    if apb.use_ada_layer_norm:\n        o = apb.sigmoid(apb.linear_ada_out(s)) * o",
         "    if _uses_ada_ln(apb):\n        o = apb.sigmoid(apb.linear_ada_out(s)) * o"),
        # FIX 2
        ("def triatt_fns(ta, use_high_precision: bool = False):",
         "def triatt_fns(ta, use_high_precision: bool = False, ending: bool = False):"),
        ("    kernel = triatt_kernel()",
         "    _bias_t = bool(ending) and _supports_transpose_bias(ta)\n    kernel = triatt_kernel()"),
        (TB_OLD, TB_NEW),
        ("bias_transposed=False)", "bias_transposed=_bias_t)"),
        ("triatt_end=triatt_fns(blk.tri_att_end),",
         "triatt_end=triatt_fns(blk.tri_att_end, ending=True),"),
    ],
    "diffusion.py": [
        # FIX 6
        ("rots = _drawn(draws, lambda: _dm.sample_rotations(shape=lead, dtype=xl.dtype, device=xl.device), chunk)",
         "rots = _drawn(draws, lambda: _sample_rotations(lead, xl.dtype, xl.device), chunk)"),
        # FIX 5
        ("        return apb.linear_z(apb.layer_norm_z(z_rows))",
         "        return apb.linear_z(_apb_ln_z(apb, z_rows))"),
        ('    if hasattr(D, "DitBias"):',
         '    if hasattr(D, "DitBias") and getattr(apb, "layer_norm_z", None) is not None:'),
        ('    D = C.seam("diffusion")\n    fns = [dit_block_fns(b, mask,',
         '    if _dt_norms_z_once(dt):                                  '
         '# OF3 0.5.0: the pair-bias LayerNorm moved here from the blocks\n'
         '        z_cond_loc = dt.layer_norm_z(z_cond_loc)\n'
         '    D = C.seam("diffusion")\n    fns = [dit_block_fns(b, mask,'),
        # FIX 3
        ("    if dt.use_cross_attention:",
         "    if _dt_uses_cross_attention(dt):"),
        # FIX 1
        ("        return apb.layer_norm_a(a, s) if apb.use_ada_layer_norm else apb.layer_norm_a(a)",
         "        return apb.layer_norm_a(a, s) if _uses_ada_ln(apb) else apb.layer_norm_a(a)"),
        ("        if apb.use_ada_layer_norm:\n            o_rows = apb.sigmoid(apb.linear_ada_out(s_rows)) * o_rows",
         "        if _uses_ada_ln(apb):\n            o_rows = apb.sigmoid(apb.linear_ada_out(s_rows)) * o_rows"),
    ],
}

HELPER_ROTS = """

def _sample_rotations(shape, dtype, device):
    \"\"\"OF3's uniform random rotations, wherever the release keeps them.

    0.4.1: structure.diffusion_module.sample_rotations
    0.5.0: structure.augmentation.sample_rotations  (same signature)
    (af3-setonix port_of3_050.py)
    \"\"\"
    # Self-contained: the adapter imports diffusion_module as `_dm` INSIDE a
    # function, so the name is not in module scope here (NameError otherwise).
    try:
        from openfold3.core.model.structure import diffusion_module as _m
        fn = getattr(_m, "sample_rotations", None)
    except ImportError:
        fn = None
    if fn is None:
        from openfold3.core.model.structure import augmentation as _aug
        fn = _aug.sample_rotations
    return fn(shape=shape, dtype=dtype, device=device)
"""

HELPERS = {"pairstack.py": (HELPER_ADA, HELPER_TB),
           "diffusion.py": (HELPER_ADA, HELPER_CROSS, HELPER_DITLN, HELPER_ROTS)}


def insert_helpers(src: str, helpers) -> str:
    add = ""
    for h in helpers:
        name = h.split("def ", 1)[1].split("(", 1)[0]
        if f"def {name}" not in src:
            add += h
    if not add:
        return src
    lines = src.split("\n")
    for i, ln in enumerate(lines):
        if ln.startswith(("def ", "class ")):
            return "\n".join(lines[:i]) + add + "\n" + "\n".join(lines[i:])
    return src + add


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=pathlib.Path)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--apply", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--revert", action="store_true")
    a = ap.parse_args()

    rc = 0
    for fname, edits in EDITS.items():
        f = a.path / fname
        if not f.is_file():
            print(f"  [MISS] {fname}")
            rc = 1
            continue
        orig = f.with_suffix(f.suffix + ".orig")

        if a.revert:
            if orig.is_file():
                shutil.copy2(orig, f)
                print(f"  [REVERT] {fname}")
            else:
                print(f"  [SKIP] {fname}: no .orig")
            continue

        src = f.read_text(encoding="utf-8", errors="surrogateescape")
        # An edit whose replacement CONTAINS its match (the _bias_t insert, the tb
        # transpose) still matches after patching -- re-applying would duplicate the
        # inserted line. Treat an edit as done when its distinguishing NEW text is
        # already present, not merely when the old text is absent.
        def _done(o, n):
            marker = n.replace(o, "").strip()
            return (marker and marker.split("\n")[0].strip() in src) or (o not in src and n in src)

        todo = [(o, n) for o, n in edits if o in src and not _done(o, n)]
        done = [(o, n) for o, n in edits if _done(o, n)]

        if a.check:
            print(f"  {fname}: {len(todo)} to patch, {len(done)}/{len(edits)} already patched")
            continue

        if not todo:
            print(f"  [OK] {fname}: already patched ({len(done)}/{len(edits)})")
            continue
        if not orig.is_file():
            shutil.copy2(f, orig)
        new = src
        for o, n in todo:
            new = new.replace(o, n)
        new = insert_helpers(new, HELPERS.get(fname, ()))
        f.write_text(new, encoding="utf-8", errors="surrogateescape")
        print(f"  [PATCH] {fname}: {len(todo)} site(s)")

    if a.apply:
        for fname in EDITS:
            f = a.path / fname
            if f.is_file():
                try:
                    ast.parse(f.read_text(encoding="utf-8", errors="replace"))
                    print(f"  [SYNTAX OK] {fname}")
                except SyntaxError as exc:
                    print(f"  [SYNTAX BROKEN] {fname}: {exc}")
                    rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
