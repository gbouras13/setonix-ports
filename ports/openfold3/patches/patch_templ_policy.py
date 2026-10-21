#!/usr/bin/env python3
'''Tell the template guard it is running OpenFold3 0.5.0, not 0.4.1.

WHY (this is a missed 0.4.1 -> 0.5.0 porting drift, the 7th)
  openfold3_opt/report.py hardcodes
      TEMPL_POLICY = ('ignore', 'upstream_0.4.1')
  so templ_guard.policy() returns ('ignore','upstream_0.4.1') and tp_rowpair/
  data.py REFUSES any query carrying real template hits:
      DataRefused: real template hits ... no template-consuming upstream fix
  That guard is correct FOR 0.4.1, where stock inference passed
  template_cache_directory=None and silently dropped templates (the bug that
  --upstream-fix OF3-001 patches). We run 0.5.0.

EVIDENCE THAT 0.5.0 CONSUMES TEMPLATES NATIVELY
  1. inference.py passes template_cache_directory=self.template_preprocessor_
     settings.cache_directory -- a real directory, not None (0.4.1 passed None).
  2. templ_guard.policy()'s own docstring names ('consume','upstream_0.5.0') as
     the other valid value, i.e. the kit anticipated this engine.
  3. The 0.5.0 kit (openfold3_ob0) ships OB0-001/OB0-002 and NO OF3-001
     equivalent -- consistent with the issue being fixed upstream in 0.5.0.

  So applying OF3-001 here would re-patch already-fixed code. The correct change
  is the policy word.

VERIFY AFTER APPLYING: a templated run must show real template slots in the
census, not all_dummy. If it does not, revert -- the premise was wrong.

USAGE  python3 patch_templ_policy.py --apply | --check | --revert
'''
import sys, pathlib
P = pathlib.Path('/scratch/pawsey1018/gbouras/af3-setonix/of3_opt_src/opt/openfold3_opt/report.py')
if len(sys.argv) > 2:
    P = pathlib.Path(sys.argv[2])
OLD = 'TEMPL_POLICY = ("ignore", "upstream_0.4.1")'
NEW = 'TEMPL_POLICY = ("consume", "upstream_0.5.0")'

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else '--check'
    s = P.read_text()
    if mode == '--check':
        print('patched (0.5.0 consume):', NEW in s)
        print('unpatched (0.4.1 ignore):', OLD in s); return 0
    if mode == '--apply':
        if NEW in s: print('already applied'); return 0
        if OLD not in s: print('ANCHOR NOT FOUND', file=sys.stderr); return 1
        P.with_suffix('.py.pre_templ_policy').write_text(s)
        P.write_text(s.replace(OLD, NEW, 1)); print('applied'); return 0
    if mode == '--revert':
        b = P.with_suffix('.py.pre_templ_policy')
        if not b.exists(): print('no backup', file=sys.stderr); return 1
        P.write_text(b.read_text()); print('reverted'); return 0
    print('usage: --apply|--check|--revert [path]', file=sys.stderr); return 2
sys.exit(main())
