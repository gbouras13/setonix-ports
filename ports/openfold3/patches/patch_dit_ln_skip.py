#!/usr/bin/env python3
"""Skip the DiT's per-step pair LayerNorm once the roll-out's pair-bias cache holds every block (env-gated; off by default).

WHY
  OpenFold3 0.5.0 normalises the conditioned pair once per DiffusionTransformer call (DiffusionTransformer.forward: layer_norm_z), and
  the tp line does the same on this rank's rows (tp_rowpair/diffusion.py diffusion_transformer_rows: z_cond_loc = dt.layer_norm_z(z_cond_loc))
  -- on EVERY denoising step, over the whole [1, 1, R, N, c_z] shard. Its only reader is the blocks' bias producer, called through the
  roll-out's PairBiasCache (opt_core diffusion_transformer_sharded: cache.get(i, lambda: pair_bias_rows(fns.bias, z_cond_shard, ...))).
  With the cache on (diff_bias_cache=on: 2,310 res / P=4 holds 2.72 GB, 4,776 hits and 24 misses per sample) every block hits from the
  second step of a sample on, so the normalised tensor is computed and never read: PyTorch's LayerNorm kernel was 27% of a denoising
  step there (~35 ms of ~130 ms; job 50323469's step kernel table), i.e. ~7 s per sample.

WHAT
  ROWPAIR_DIT_LN_SKIP=1: when the roll-out's cache is enabled and holds a bias for every block, the LayerNorm is not run and the
  un-normalised shard is passed on (diffusion_transformer_sharded only checks its shape; no producer runs). Otherwise -- the first step of
  a sample, the cache off (large N), or any block missing -- the statement runs exactly as before. Nothing is computed differently, so
  outputs are bitwise those without it. Unset = the original statement plus nothing (the gate is read once at import).

USAGE  python3 patch_dit_ln_skip.py --apply | --check | --revert [path]
"""
import os, sys, pathlib

P = pathlib.Path('/scratch/pawsey1018/gbouras/af3-setonix/of3_opt_src/opt/openfold3_opt/tp_rowpair/diffusion.py')
if len(sys.argv) > 2:
    P = pathlib.Path(sys.argv[2])
MARK = '# [dit_ln_skip]'
BACKUP = P.with_suffix('.py.pre_dit_ln_skip')

OLD_STMT = ("    if _dt_norms_z_once(dt):                                  # OF3 0.5.0: the pair-bias LayerNorm moved here from the blocks\n"
            "        z_cond_loc = dt.layer_norm_z(z_cond_loc)\n")
NEW_STMT = ("    if _dt_norms_z_once(dt):                                  # OF3 0.5.0: the pair-bias LayerNorm moved here from the blocks\n"
            "        _bc = getattr(cache, \"bias_cache\", None)              # [dit_ln_skip] the normalised z feeds only the blocks' bias producers;\n"
            "        if not (_DIT_LN_SKIP and _bc is not None and _bc.enabled and len(dt.blocks) > 0       # when the roll-out's cache holds every\n"
            "                and all(i in _bc.store for i in range(len(dt.blocks)))):                     # block's bias none of them runs, so the\n"
            "            z_cond_loc = dt.layer_norm_z(z_cond_loc)          # LayerNorm is skipped (its output would be unread)\n")
OLD_ANCHOR = "def diffusion_transformer_rows(dt, a, s, z_cond_loc, mask, cache: RolloutCache, use_high_precision_attention=False, _mask_trans=True, **flags):\n"
GATE = ("# [dit_ln_skip] ROWPAIR_DIT_LN_SKIP=1: skip the per-step pair LayerNorm below once the roll-out's PairBiasCache holds every block\n"
        "# (its only reader); bitwise unchanged. Unset = the statement as before.\n"
        "_DIT_LN_SKIP = os.environ.get(\"ROWPAIR_DIT_LN_SKIP\", \"\") == \"1\"\n"
        "\n"
        "\n")


def _apply(s):
    counts = {"stmt": s.count(OLD_STMT), "anchor": s.count(OLD_ANCHOR)}
    if any(n != 1 for n in counts.values()):
        raise SystemExit(f'ANCHORS NOT FOUND EXACTLY ONCE ({counts}) -- not patching')
    if '\nimport os\n' not in s:
        raise SystemExit('diffusion.py lacks "import os" at module level -- not patching')
    new = s.replace(OLD_ANCHOR, GATE + OLD_ANCHOR, 1).replace(OLD_STMT, NEW_STMT, 1)
    compile(new, str(P), 'exec')
    return new


def _write(text):
    tmp = P.with_suffix('.py.dit_ln_skip_tmp')
    tmp.write_text(text)
    os.replace(tmp, P)                                     # atomic: importers never see a half-written file


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else '--check'
    s = P.read_text()
    applied = MARK in s
    if mode == '--check':
        print(f'dit_ln_skip: applied={applied}  backup={BACKUP.exists()}')
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
