#!/usr/bin/env python3
"""Say which path each triangle attention takes to the core, and in which dtype (env-gated diagnostic).

WHY
  With ROWPAIR_TRIATT_OPERANDS=fp16 the fused kernel serves the trunk (1,296 calls on the hexamer), but 120 calls still
  reach it in fp32 and fall back to the eager core (fallback_by=unsupported:dtype_float32:120, jobs 50030499, 50071993).
  120 is exactly the confidence head's count (4 blocks x 2 nodes x 5 samples x 3 row blocks), and the confidence time
  did not move (14.2 -> 13.9 s per sample). By the code, the confidence pairformer binds the same triatt_fns.attend
  that casts, so the path those calls take is not known. This prints it.

WHAT
  ROWPAIR_TRIATT_DIAG=1 (rank 0 only). For each distinct (path, node, x_rows shape/dtype, q dtype) the FIRST call prints
  one line with a short call stack. The paths are:
    attend   triatt_fns.attend, the whole-rows statement where the fp16 cast sits (q as it goes to the dispatch);
    proj     triatt_fns.proj, a piece the core's just-in-time bias schedule assembles itself;
    core     the dispatch as handed to the core (TriAttFns.core), with the dtype q actually has there.
  At exit, one summary line of call counts per key. Unset = the same statements as before plus one bool test per call.
  Leaves the [triatt_operands v2] lines untouched (the job scripts' startup checks grep for them).

USAGE  python3 patch_triatt_diag.py --apply | --check | --revert [path]
"""
import os, sys, pathlib

P = pathlib.Path('/scratch/pawsey1018/gbouras/af3-setonix/of3_opt_src/opt/openfold3_opt/tp_rowpair/pairstack.py')
if len(sys.argv) > 2:
    P = pathlib.Path(sys.argv[2])
MARK = '# [triatt_diag]'
BACKUP = P.with_suffix('.py.pre_triatt_diag')

OLD_DEF = "def triatt_fns(ta, use_high_precision: bool = False, ending: bool = False):\n"
HELPERS = '''# [triatt_diag] ROWPAIR_TRIATT_DIAG=1 (rank 0): which path each triangle attention takes to the core, in which dtype.
_TRIATT_DIAG = (os.environ.get("ROWPAIR_TRIATT_DIAG", "") == "1"
                and (os.environ.get("ROWPAIR_RANK") or os.environ.get("OF3TP_RANK") or "0").strip() == "0")
_TRIATT_DIAG_SEEN: dict = {}


def _triatt_diag(where, ending, x=None, q=None, k=None):                               # [triatt_diag]
    key = (where, bool(ending), None if x is None else tuple(x.shape[-2:]), None if x is None else str(x.dtype),
           None if q is None else str(q.dtype))
    n = _TRIATT_DIAG_SEEN.get(key, 0) + 1
    _TRIATT_DIAG_SEEN[key] = n
    if n == 1:
        import traceback
        stack = " < ".join(f"{f.name}:{f.lineno}" for f in traceback.extract_stack(limit=12)[:-1][::-1])
        xs = "-" if x is None else f"{tuple(x.shape)} {x.dtype}"
        qs = "-" if q is None else f"{tuple(q.shape)} {q.dtype}"
        ks = "-" if k is None else str(int(k.shape[-2]))
        print(f"[triatt_diag] first {where} ending={bool(ending)} x={xs} q={qs} keys={ks} "
              f"OPERANDS={os.environ.get('ROWPAIR_TRIATT_OPERANDS', '')!r} | {stack}", flush=True)


def _triatt_diag_core(dispatch, ending):                                               # [triatt_diag]
    def core(q, k, v, *a, **kw):
        _triatt_diag("core", ending, q=q, k=k)
        return dispatch(q, k, v, *a, **kw)
    return core


def _triatt_diag_exit():                                                               # [triatt_diag]
    if _TRIATT_DIAG_SEEN:
        print("[triatt_diag] calls: " + "; ".join(f"{k[0]} ending={int(k[1])} x={k[2]} {k[3]} q={k[4]}: {n}"
                                                  for k, n in sorted(_TRIATT_DIAG_SEEN.items(), key=str)), flush=True)


if _TRIATT_DIAG:
    atexit.register(_triatt_diag_exit)


'''
OLD_CALL = "        o = dispatch(q, k, v, [mask_bias, tb])\n"
NEW_CALL = ("        if _TRIATT_DIAG:                                                               # [triatt_diag]\n"
            "            _triatt_diag(\"attend\", ending, x=x_rows, q=q, k=k)\n"
            "        o = dispatch(q, k, v, [mask_bias, tb])\n")
OLD_PROJ = "    def proj(x_rows):\n        return ta.mha._prep_qkv(x_rows, x_rows)\n"
NEW_PROJ = ("    def proj(x_rows):\n"
            "        if _TRIATT_DIAG:                                                               # [triatt_diag]\n"
            "            _triatt_diag(\"proj\", ending, x=x_rows)\n"
            "        return ta.mha._prep_qkv(x_rows, x_rows)\n")
OLD_CORE = "core=dispatch, bias_transposed=_bias_t)"
NEW_CORE = "core=(_triatt_diag_core(dispatch, ending) if _TRIATT_DIAG else dispatch), bias_transposed=_bias_t)"


def _apply(s):
    counts = {name: s.count(a) for name, a in (("def", OLD_DEF), ("call", OLD_CALL), ("proj", OLD_PROJ), ("core", OLD_CORE))}
    if any(n != 1 for n in counts.values()):
        raise SystemExit(f'ANCHORS NOT FOUND EXACTLY ONCE ({counts}) -- not patching')
    for need in ('import atexit', 'import os'):
        if need not in s:
            raise SystemExit(f'pairstack.py lacks "{need}" at module level -- not patching')
    new = (s.replace(OLD_DEF, HELPERS + OLD_DEF, 1).replace(OLD_CALL, NEW_CALL, 1)
            .replace(OLD_PROJ, NEW_PROJ, 1).replace(OLD_CORE, NEW_CORE, 1))
    compile(new, str(P), 'exec')
    return new


def _write(text):
    tmp = P.with_suffix('.py.triatt_diag_tmp')
    tmp.write_text(text)
    os.replace(tmp, P)                                     # atomic: importers never see a half-written file


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else '--check'
    s = P.read_text()
    applied = MARK in s
    if mode == '--check':
        print(f'triatt_diag: applied={applied}  backup={BACKUP.exists()}  v2_marker={"[triatt_operands v2]" in s}')
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
