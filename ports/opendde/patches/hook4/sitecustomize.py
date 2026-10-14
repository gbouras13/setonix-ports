"""sitecustomize for v2 + the chunk floor + explicit triangle-attention blocks (patches/hook4 first on PYTHONPATH).

Runs patches/hook3/sitecustomize.py unchanged -- hook2's per-rank cache hook and v2 (ODDE_TRUNK_PATCH=1), then the
chunk floor (ODDE_CHUNK_FLOOR=1) -- and then, only when ODDE_TP_BLOCKS=1, arms the two kit-seam wrappers of
patches/odde_tp_blocks_patch.py (opendde_opt.tp._attn_rows, opendde_opt.tp_kernels.attention_dispatch) through the
kit's own opt_core.autoload.patch_attr_at_import. The validated v2 and floor paths are untouched: this is a
separate hook directory and a separate flag. Fails closed (exit 3) if the blocks cannot be armed.
"""
import os
import runpy
import sys

_ROOT = os.environ.get("ODDE_ROOT", "/scratch/pawsey1018/gbouras/opendde-setonix")
_H3 = os.path.join(_ROOT, "patches", "hook3", "sitecustomize.py")
if os.path.isfile(_H3):
    runpy.run_path(_H3)

if os.environ.get("ODDE_TP_BLOCKS") == "1":
    _PATCHES = os.path.join(_ROOT, "patches")
    if _PATCHES not in sys.path:
        sys.path.insert(0, _PATCHES)
    try:
        import odde_tp_blocks_patch as _tb
        from opt_core import autoload as _al
        _tb.rows()
        _tb.qblock()                      # refuse a bad block size before any rank starts
        _al.patch_attr_at_import("opendde_opt.tp", "_attn_rows", _tb.make_attn_rows_logged,
                                 tag="odde-tp-blocks", name="attn_rows")
        _al.patch_attr_at_import("opendde_opt.tp_kernels", "attention_dispatch", _tb.make_attention_dispatch_logged,
                                 tag="odde-tp-blocks", name="attention_dispatch")
    except Exception as _e:
        sys.stderr.write("[odde-tp-blocks] NOT INSTALLED: %s: %s\n" % (type(_e).__name__, _e))
        sys.stderr.flush()
        os._exit(3)
