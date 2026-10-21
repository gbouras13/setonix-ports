#!/usr/bin/env python3
'''Make RDKit conformer generation reproducible, opt-in.

WHY
  openfold3/core/data/primitives/structure/conformer.py:135 does

      strategy.randomSeed = random.randint(0, 10**9)

  deliberately (its comment explains RDKit's internal default seed). Reference
  conformers feed the ref_* features, so EVERY RUN gets different ref_pos ->
  different input features -> the trunk diverges from cycle0 -> different
  structures. Measured on the trimer, same config and seed, three runs:
      feature digest  5606fb5e34d50086 / ca3b1fdb2d5a1efd / 3605a22b18b0919b
      cycle0..3 DIFFER; samples 0.3 A to 27 A apart
  This is upstream behaviour and would affect a stock single-GPU run the same
  way. It is not sharding, kernels, or the 0.4.1->0.5.0 port.

WHAT THIS CHANGES
  Nothing by default. With OF3_CONFORMER_SEED set to an integer, that value is
  used instead of the random draw, which makes featurization -- and therefore the
  whole pipeline -- reproducible. Conformer embedding for standard residues is a
  local optimisation, so a fixed seed is scientifically cheap; leaving it unset
  preserves upstream behaviour exactly.

USAGE  python3 patch_conformer_seed.py --apply | --check | --revert
'''
import sys, pathlib
P = pathlib.Path('/scratch/pawsey1018/gbouras/af3-setonix/openfold3/openfold3/core/data/primitives/structure/conformer.py')
if len(sys.argv) > 2:
    P = pathlib.Path(sys.argv[2])
OLD = '    strategy.randomSeed = random.randint(0, 10**9)'
NEW = ('    # [of3-setonix] OF3_CONFORMER_SEED=<int> pins this draw so ref_* features, and thus\n'
       '    # the whole pipeline, are reproducible run to run. Unset = upstream behaviour.\n'
       '    import os as _os\n'
       '    _cseed = (_os.environ.get("OF3_CONFORMER_SEED") or "").strip()\n'
       '    strategy.randomSeed = int(_cseed) if _cseed.lstrip("-").isdigit() else random.randint(0, 10**9)')

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else '--check'
    s = P.read_text()
    applied = 'OF3_CONFORMER_SEED' in s
    if mode == '--check':
        print('conformer-seed patch applied:', applied); print('anchor present:', OLD in s); return 0
    if mode == '--apply':
        if applied: print('already applied'); return 0
        if OLD not in s: print('ANCHOR NOT FOUND', file=sys.stderr); return 1
        P.with_suffix('.py.pre_conformer_seed').write_text(s)
        P.write_text(s.replace(OLD, NEW, 1)); print('applied'); return 0
    if mode == '--revert':
        b = P.with_suffix('.py.pre_conformer_seed')
        if not b.exists(): print('no backup', file=sys.stderr); return 1
        P.write_text(b.read_text()); print('reverted'); return 0
    print('usage: --apply|--check|--revert [path]', file=sys.stderr); return 2
sys.exit(main())
