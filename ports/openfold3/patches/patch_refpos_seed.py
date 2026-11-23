#!/usr/bin/env python3
'''Pin ref_pos so kernel numerics can be measured, opt-in.

WHY. featurization/conformer.py:149 applies centre_random_augmentation() to the
reference conformer coordinates -- a random rotation + translation from
torch.randn, by AF3 design so the model cannot learn absolute orientation. It is
the ONE input tensor that differs between two runs with identical config and
seed (measured: 43 of 44 per-key digests match, ref_pos differs). Because it is
upstream of the trunk, EVERY structural comparison across runs is dominated by
it -- which is why the kernel factorial's 30-42 A diffs and the retune's 2.4-7.0
A diffs measured nothing about kernels.

WHAT THIS CHANGES. Nothing by default. With OF3_REF_POS_SEED=<int> the torch
global RNG is seeded immediately before the augmentation loop, so ref_pos is
identical run to run and a cross-run comparison isolates the kernels. Unset =
upstream behaviour, augmentation untouched.

USAGE  python3 patch_refpos_seed.py --apply | --check | --revert
'''
import sys, pathlib
P = pathlib.Path('/scratch/pawsey1018/gbouras/af3-setonix/openfold3/openfold3/core/data/pipelines/featurization/conformer.py')
if len(sys.argv) > 2:
    P = pathlib.Path(sys.argv[2])
OLD = '''        if torch.any(mol_ref_mask):
            # Apply random translation & rotation to reference coordinates
            final_ref_pos = centre_random_augmentation(mol_ref_pos, mol_ref_mask)'''
NEW = '''        if torch.any(mol_ref_mask):
            # [of3-setonix] OF3_REF_POS_SEED=<int> pins this augmentation so ref_pos is
            # reproducible and cross-run comparisons isolate the model, not the input.
            # Unset = upstream behaviour (a fresh random rotation+translation per call).
            import os as _os
            _rs = (_os.environ.get("OF3_REF_POS_SEED") or "").strip()
            if _rs.lstrip("-").isdigit():
                torch.manual_seed(int(_rs) + mol_idx)
            # Apply random translation & rotation to reference coordinates
            final_ref_pos = centre_random_augmentation(mol_ref_pos, mol_ref_mask)'''

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else '--check'
    s = P.read_text(); applied = 'OF3_REF_POS_SEED' in s
    if mode == '--check':
        print('refpos patch applied:', applied); print('anchor present:', OLD in s); return 0
    if mode == '--apply':
        if applied: print('already applied'); return 0
        if OLD not in s: print('ANCHOR NOT FOUND', file=sys.stderr); return 1
        P.with_suffix('.py.pre_refpos').write_text(s); P.write_text(s.replace(OLD, NEW, 1)); print('applied'); return 0
    if mode == '--revert':
        b = P.with_suffix('.py.pre_refpos')
        if not b.exists(): print('no backup', file=sys.stderr); return 1
        P.write_text(b.read_text()); print('reverted'); return 0
    print('usage: --apply|--check|--revert [path]', file=sys.stderr); return 2
sys.exit(main())
