"""sitecustomize for everything in hook6 plus row-blocked APB pair bias (patches/hook7 first on PYTHONPATH).

Runs patches/hook6/sitecustomize.py unchanged (hook5 -> hook4 -> hook3 -> hook2 -> the base per-rank hook, each gated by
its own flag; hook6 adds ODDE_ATTN_QBLOCK). Then, only when ODDE_APB_ROWS_PATCH=1, it arms patches/odde_apb_rows_patch.py
on opendde_opt.tp._apb_rows_of through the kit's own opt_core.autoload.patch_attr_at_import. The validated paths are
untouched: this is a separate hook directory and a separate flag. Fails closed (exit 3) if it cannot be armed, a size is
invalid, or the kit's _apb_rows_of source is not the one the patch was written against.
"""
import os
import runpy
import sys

_ROOT = os.environ.get("ODDE_ROOT", "/scratch/pawsey1018/gbouras/opendde-setonix")
_H6 = os.path.join(_ROOT, "patches", "hook6", "sitecustomize.py")
if os.path.isfile(_H6):
    runpy.run_path(_H6)

if os.environ.get("ODDE_APB_ROWS_PATCH") == "1":
    _PATCHES = os.path.join(_ROOT, "patches")
    if _PATCHES not in sys.path:
        sys.path.insert(0, _PATCHES)
    try:
        import odde_apb_rows_patch as _ap
        from opt_core import autoload as _al
        _ap.rows()
        _ap.min_gib()                     # refuse a bad size before anything runs

        def _make(orig):
            try:
                return _ap.make_apb_rows_of_logged(orig)
            except Exception as e:        # the source guard fires at import of opendde_opt.tp: fail closed there too
                sys.stderr.write("[odde-apb-rows] NOT INSTALLED: %s: %s\n" % (type(e).__name__, e))
                sys.stderr.flush()
                os._exit(3)
        _al.patch_attr_at_import("opendde_opt.tp", "_apb_rows_of", _make, tag="odde-apb-rows", name="_apb_rows_of")
    except Exception as _e:
        sys.stderr.write("[odde-apb-rows] NOT INSTALLED: %s: %s\n" % (type(_e).__name__, _e))
        sys.stderr.flush()
        os._exit(3)
