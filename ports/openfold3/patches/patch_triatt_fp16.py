#!/usr/bin/env python3
"""Feed triangle attention's core 16-bit operands, so opt_core's fused flash_triattn can serve it on gfx90a  (v2).

WHY
  OpenFold3's pair stack is fp32 and gfx90a has no TF32, so the trunk's triangle-attention GEMMs run at the plain fp32
  rate (an H100 silently uses TF32 under float32_matmul_precision="high"). opt_core's fused flash_triattn builds on
  gfx90a once OF3T_FLASH_IP=ieee (the default 'tf32' is not an AMD precision). Through the kit's own dispatch (job
  50059185) its fp16 path is 6.2-9.4x faster than today's eager fp32 core at N = 6,930 / 12,705 / 15,400 with 0.9-1.3e-3
  relative error, while its fp32 path (ieee or bf16x3) is finite but 40-65% WRONG from N = 12,705 up. The kit's dispatch
  serves bf16/fp16 by default (serve_dtypes), so the missing piece is 16-bit operands.

WHAT
  ROWPAIR_TRIATT_OPERANDS=fp16 (or bf16) casts q/k/v to that dtype in triatt_fns.attend before the dispatch and casts
  the output back to the pair stack's dtype before the stock gate/output projection. The biases (key mask, triangle
  bias) stay fp32: the flash serve layer prepares its own bias copies from an fp32 bias. Unset = byte-for-byte today.

  v2 SIZE GUARD. The fp32 failure appears where an fp32 bias plane passes 2**31 BYTES (16*N^2 at H=4: N ~ 11,585); the
  16-bit plane passes it at N ~ 16,384, just above the largest size verified correct (15,400). So the cast is applied
  only for N <= ROWPAIR_TRIATT_FP16_MAX_N (default 15400, the largest verified). Above it the operands stay fp32, which
  the dispatch does not serve (serve_dtypes), so the call falls back BY NAME to the exact eager core -- correct, slower.
  A one-line note is printed the first time a plane is kept fp32. Raise the limit only after verifying larger N.

  Use with ROWPAIR_TRIATT_CORE=flash_triattn and OF3T_FLASH_IP=ieee (the scripts/predict-sharded.sh defaults).

USAGE  python3 patch_triatt_fp16.py --apply | --check | --revert [path]      (--apply upgrades an applied v1 in place)
"""
import os, sys, pathlib

P = pathlib.Path('/scratch/pawsey1018/gbouras/af3-setonix/of3_opt_src/opt/openfold3_opt/tp_rowpair/pairstack.py')
if len(sys.argv) > 2:
    P = pathlib.Path(sys.argv[2])
MARK_V1 = '# [triatt_operands]'
MARK_V2 = '# [triatt_operands v2]'
BACKUP = P.with_suffix('.py.pre_triatt_fp16')             # the unpatched original (written by v1 or v2, whichever applied first)

OLD_BIND = "    _register_exit_lines()\n\n    def ln(z_rows):\n        return ta.layer_norm(z_rows)\n"
NEW_BIND = ("    _register_exit_lines()\n"
            "    # [triatt_operands v2] ROWPAIR_TRIATT_OPERANDS=fp16|bf16: q/k/v cast to 16-bit before the core's dispatch (the biases\n"
            "    # stay fp32: the flash serve layer prepares its own bias copies), the output cast back to the pair stack's dtype.\n"
            "    # gfx90a has no TF32, so fp32 GEMMs run at 1/4 of the 16-bit rate; fp16's 10-bit mantissa is TF32's class.\n"
            "    # Only for N <= ROWPAIR_TRIATT_FP16_MAX_N (default 15400, the largest size verified correct through the dispatch):\n"
            "    # the kernel's fp32 path goes wrong once a bias plane passes 2**31 bytes, and a 16-bit plane passes it at N ~ 16,384.\n"
            "    # Above the limit the operands stay fp32 -> not served -> the exact eager core, by name. Unset = fp32 (stock).\n"
            "    _opd = {\"fp16\": torch.float16, \"bf16\": torch.bfloat16}.get(os.environ.get(\"ROWPAIR_TRIATT_OPERANDS\", \"\").strip().lower())\n"
            "    _opd_max_n = int(os.environ.get(\"ROWPAIR_TRIATT_FP16_MAX_N\", \"15400\") or 15400)\n"
            "\n    def ln(z_rows):\n        return ta.layer_norm(z_rows)\n")
OLD_CALL = "        o = dispatch(q, k, v, [mask_bias, tb])\n"
NEW_CALL = ("        if _opd is not None:                                                           # [triatt_operands v2]\n"
            "            if int(x_rows.shape[-2]) <= _opd_max_n:\n"
            "                q, k, v = q.to(_opd), k.to(_opd), v.to(_opd)\n"
            "            elif not TRIATT.get(\"_opd_note\"):\n"
            "                TRIATT[\"_opd_note\"] = True\n"
            "                print(f\"[tp_rowpair] triatt operands kept fp32 at N={int(x_rows.shape[-2])} > ROWPAIR_TRIATT_FP16_MAX_N=\"\n"
            "                      f\"{_opd_max_n}: the fused kernel is unverified there; the exact eager core serves\", flush=True)\n"
            "        o = dispatch(q, k, v, [mask_bias, tb])\n"
            "        if o.dtype != x_rows.dtype:\n"
            "            o = o.to(x_rows.dtype)\n")


def _apply_v2(s):
    if s.count(OLD_BIND) != 1 or s.count(OLD_CALL) != 1:
        raise SystemExit(f'ANCHORS NOT FOUND EXACTLY ONCE (bind={s.count(OLD_BIND)} call={s.count(OLD_CALL)}) -- not patching')
    for need in ('import torch', 'import os'):
        if need not in s:
            raise SystemExit(f'pairstack.py lacks "{need}" at module level -- not patching')
    new = s.replace(OLD_BIND, NEW_BIND, 1).replace(OLD_CALL, NEW_CALL, 1)
    compile(new, str(P), 'exec')
    return new


def _write(text):
    tmp = P.with_suffix('.py.triatt_fp16_tmp')
    tmp.write_text(text)
    os.replace(tmp, P)                                     # atomic: importers never see a half-written file


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else '--check'
    s = P.read_text()
    v2, v1 = MARK_V2 in s, (MARK_V1 in s and MARK_V2 not in s)
    if mode == '--check':
        print(f'triatt_fp16: v2 applied={v2}  v1 applied={v1}  backup={BACKUP.exists()}')
        return 0
    if mode == '--apply':
        if v2:
            print('v2 already applied'); return 0
        if v1:                                             # upgrade: rebuild from the unpatched original
            if not BACKUP.exists():
                print('v1 applied but no backup to rebuild from -- not patching', file=sys.stderr); return 1
            _write(_apply_v2(BACKUP.read_text()))
            print('upgraded v1 -> v2 (rebuilt from the unpatched backup)'); return 0
        BACKUP.write_text(s)
        _write(_apply_v2(s))
        print(f'applied v2 (backup: {BACKUP.name})'); return 0
    if mode == '--revert':
        if not BACKUP.exists():
            print('no backup', file=sys.stderr); return 1
        _write(BACKUP.read_text())
        print('reverted'); return 0
    print('usage: --apply | --check | --revert [path]', file=sys.stderr); return 2


sys.exit(main())
