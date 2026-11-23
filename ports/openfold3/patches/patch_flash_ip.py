#!/usr/bin/env python3
"""Let flash_triattn's fp32 input precision be one AMD Triton accepts: bf16x3 / bf16x6 (in addition to tf32/tf32x3/ieee).

WHY
  opt_core/kernels/flash_triattn.py reads OF3T_FLASH_IP (default 'tf32') and asserts it is one of tf32 / tf32x3 / ieee.
  Triton's AMD backend accepts only ieee / bf16x3 / bf16x6 (job 49796402: "input_precision must be one of ('ieee',
  'bf16x3', 'bf16x6'). Got tf32"). 'ieee' builds but is true fp32; 'bf16x3' is the 3-pass bf16 emulation of fp32 --
  AMD's closest analogue to what TF32x3 is on NVIDIA -- measured 6e-6 rel. error at 6,930 tokens in a direct call.
  This only widens the accepted set; the default ('tf32') and every existing value behave exactly as before.

USAGE  python3 patch_flash_ip.py --apply | --check | --revert [path]
"""
import os, sys, pathlib
P = pathlib.Path('/scratch/pawsey1018/gbouras/af3-setonix/opt_core_src/opt_core/opt_core/kernels/flash_triattn.py')
if len(sys.argv) > 2:
    P = pathlib.Path(sys.argv[2])
MARK = '# [flash_ip]'
BACKUP = P.with_suffix('.py.pre_flash_ip')
OLD = 'assert _IP_DEFAULT in ("tf32", "tf32x3", "ieee"), _IP_DEFAULT\n'
NEW = ('assert _IP_DEFAULT in ("tf32", "tf32x3", "ieee", "bf16x3", "bf16x6"), _IP_DEFAULT   '
       '# [flash_ip] bf16x3/bf16x6: the fp32 precisions AMD Triton accepts besides ieee\n')

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else '--check'
    s = P.read_text(); applied = MARK in s
    if mode == '--check':
        print(f'flash_ip patch applied: {applied}; anchor present: {s.count(OLD) == 1}'); return 0
    if mode == '--apply':
        if applied:
            print('already applied'); return 0
        if s.count(OLD) != 1:
            print(f'ANCHOR NOT FOUND EXACTLY ONCE ({s.count(OLD)}) -- not patching', file=sys.stderr); return 1
        new = s.replace(OLD, NEW, 1); compile(new, str(P), 'exec')
        BACKUP.write_text(s); tmp = P.with_suffix('.py.flash_ip_tmp'); tmp.write_text(new); os.replace(tmp, P)
        print(f'applied (backup: {BACKUP.name})'); return 0
    if mode == '--revert':
        if not BACKUP.exists():
            print('no backup', file=sys.stderr); return 1
        tmp = P.with_suffix('.py.flash_ip_tmp'); tmp.write_text(BACKUP.read_text()); os.replace(tmp, P)
        print('reverted'); return 0
    print('usage: --apply | --check | --revert [path]', file=sys.stderr); return 2

sys.exit(main())
