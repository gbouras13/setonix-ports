#!/usr/bin/env python3
'''Log a per-key digest of the input features, so a run-to-run difference names
the tensor instead of being guessed at.

WHY. The A/A test shows the input feature digest differs between two runs with
identical config and seed (trunk diverges from cycle0). Three hypotheses were
tested and all were wrong: PYTHONHASHSEED (hash ordering), the MSA (it is
excluded from the digest anyway), and the RDKit conformer seed
(OF3_CONFORMER_SEED=42 pinned it; digests still differ). This stops the guessing
and prints one line per feature key so a diff of two runs is decisive.

Set ROWPAIR_PERKEY_DIGEST=1 to enable. Off by default.
'''
import sys, pathlib
P = pathlib.Path('/scratch/pawsey1018/gbouras/af3-setonix/of3_opt_src/opt/openfold3_opt/tp_rowpair/model.py')
if len(sys.argv) > 2:
    P = pathlib.Path(sys.argv[2])
OLD = '    shared = fd.digest\n'
NEW = ('    shared = fd.digest\n'
       '    if os.environ.get("ROWPAIR_PERKEY_DIGEST", "") == "1":   # [perkey] name the differing tensor instead of guessing\n'
       '        import hashlib as _hl\n'
       '        try:\n'
       '            import torch as _t\n'
       '            for _k in sorted(batch):\n'
       '                _v = batch[_k]\n'
       '                if isinstance(_v, _t.Tensor):\n'
       '                    _b = _v.detach().to("cpu").contiguous().numpy().tobytes()\n'
       '                    _h = _hl.sha256(_b).hexdigest()[:16]\n'
       '                    comm.log(f"[perkey] {_k} {tuple(_v.shape)} {_v.dtype} {_h}")\n'
       '        except Exception as _e:\n'
       '            comm.log(f"[perkey] failed: {type(_e).__name__}: {_e}")\n')

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else '--check'
    s = P.read_text()
    applied = '[perkey]' in s
    if mode == '--check':
        print('perkey patch applied:', applied); print('anchor present:', OLD in s); return 0
    if mode == '--apply':
        if applied: print('already applied'); return 0
        if OLD not in s: print('ANCHOR NOT FOUND', file=sys.stderr); return 1
        P.with_suffix('.py.pre_perkey').write_text(s)
        P.write_text(s.replace(OLD, NEW, 1)); print('applied'); return 0
    if mode == '--revert':
        b = P.with_suffix('.py.pre_perkey')
        if not b.exists(): print('no backup', file=sys.stderr); return 1
        P.write_text(b.read_text()); print('reverted'); return 0
    print('usage: --apply|--check|--revert [path]', file=sys.stderr); return 2
sys.exit(main())
