#!/usr/bin/env python3
"""Let flash_triattn serve OpenFold3's fp32 pair stack.

WHY
  tp_rowpair/pairstack.py calls opt_core's attention_core() without
  serve_dtypes, so it takes the default ('bfloat16','float16'). OpenFold3's pair
  stack runs fp32, so EVERY call is refused with
  'all_fallback:unsupported:dtype_float32' and triangle attention runs the torch
  statement -- measured: 1,416 of 1,416 calls fell back, on both the trimer and
  the hexamer.

  opt_core documents the fix: 'an engine whose pair stack runs fp32 passes
  (\"bfloat16\", \"float16\", \"float32\") -- the flash kernel then computes its
  products at input_precision = tf32 unless kernel_kwargs say otherwise.'

RISK, WHICH IS WHY THIS IS ENV-GATED AND DEFAULTS OFF
  tf32 is an NVIDIA tensor-core format with no gfx90a equivalent; what Triton's
  ROCm backend does with that request is unverified. It also bypasses the
  engine's own use_high_precision path. So the gate defaults to the current
  behaviour and only ROWPAIR_TRIATT_SERVE_FP32=1 opts in.

USAGE  python3 patch_serve_fp32.py --apply | --check | --revert
"""
import re, sys, pathlib

P = pathlib.Path('/scratch/pawsey1018/gbouras/af3-setonix/of3_opt_src/opt/openfold3_opt/tp_rowpair/pairstack.py')
if len(sys.argv) > 2:
    P = pathlib.Path(sys.argv[2])
MARK = '# [serve_fp32]'

OLD = '''        dispatch = C.fn("triatt", "attention_core")(lambda q_, k_, v_, b: core(q_, k_, v_, b, use_high_precision=use_high_precision), kernel=kernel, scale=1.0,
                                                   layout="bnhsd", mask_from="bias0", tri_bias="bias1", stock_qblock=lever_rows("TRIATT_QBLOCK"))'''

NEW = '''        # [serve_fp32] OpenFold3's pair stack is fp32; the default serve_dtypes is
        # bf16/fp16, so every call is refused unsupported:dtype_float32 and triangle
        # attention never leaves the torch statement. opt_core's documented opt-in for
        # an fp32 engine is to name float32 here (the kernel then computes at
        # input_precision=tf32). Gated: tf32 has no gfx90a equivalent, so this stays
        # off until measured. ROWPAIR_TRIATT_SERVE_FP32=1 turns it on.
        _serve = ("bfloat16", "float16", "float32") if os.environ.get("ROWPAIR_TRIATT_SERVE_FP32", "") == "1" else ("bfloat16", "float16")
        dispatch = C.fn("triatt", "attention_core")(lambda q_, k_, v_, b: core(q_, k_, v_, b, use_high_precision=use_high_precision), kernel=kernel, scale=1.0,
                                                   layout="bnhsd", mask_from="bias0", tri_bias="bias1", stock_qblock=lever_rows("TRIATT_QBLOCK"),
                                                   serve_dtypes=_serve)'''

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else '--check'
    s = P.read_text()
    applied = MARK in s
    if mode == '--check':
        print(f'serve_fp32 patch applied: {applied}')
        print(f'anchor present (unpatched form): {OLD in s}')
        return 0
    if mode == '--apply':
        if applied:
            print('already applied'); return 0
        if OLD not in s:
            print('ANCHOR NOT FOUND -- upstream changed; not patching', file=sys.stderr); return 1
        P.with_suffix('.py.pre_serve_fp32').write_text(s)
        P.write_text(s.replace(OLD, NEW, 1))
        print('applied (backup: pairstack.py.pre_serve_fp32)'); return 0
    if mode == '--revert':
        b = P.with_suffix('.py.pre_serve_fp32')
        if not b.exists():
            print('no backup', file=sys.stderr); return 1
        P.write_text(b.read_text()); print('reverted'); return 0
    print('usage: --apply | --check | --revert [path]', file=sys.stderr); return 2

sys.exit(main())
