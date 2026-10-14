"""sitecustomize for v2 + explicit blocks + the native allocator + the A-block threshold + query-blocked stock attention
(patches/hook6 first on PYTHONPATH).

Runs patches/hook5/sitecustomize.py unchanged (hook4 -> hook3 -> hook2 -> the base per-rank hook, each gated by its own
flag). Then, only when ODDE_ATTN_QBLOCK=1, it arms patches/odde_attn_qblock_patch.py on
opendde.model.triangular.layers._attention through the kit's own opt_core.autoload.patch_attr_at_import. The validated
paths are untouched: this is a separate hook directory and a separate flag. Fails closed (exit 3) if it cannot be armed
or a size is invalid.
"""
import os
import runpy
import sys

_ROOT = os.environ.get("ODDE_ROOT", "/scratch/pawsey1018/gbouras/opendde-setonix")
_H5 = os.path.join(_ROOT, "patches", "hook5", "sitecustomize.py")
if os.path.isfile(_H5):
    runpy.run_path(_H5)

if os.environ.get("ODDE_ATTN_QBLOCK") == "1":
    _PATCHES = os.path.join(_ROOT, "patches")
    if _PATCHES not in sys.path:
        sys.path.insert(0, _PATCHES)
    try:
        import odde_attn_qblock_patch as _aq
        from opt_core import autoload as _al
        _aq.block_gib()                   # refuse a bad size before anything runs
        _al.patch_attr_at_import("opendde.model.triangular.layers", "_attention", _aq.make_attention_logged,
                                 tag="odde-attn-qblock", name="_attention")
    except Exception as _e:
        sys.stderr.write("[odde-attn-qblock] NOT INSTALLED: %s: %s\n" % (type(_e).__name__, _e))
        sys.stderr.flush()
        os._exit(3)
