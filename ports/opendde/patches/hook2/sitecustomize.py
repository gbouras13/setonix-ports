"""sitecustomize for the v2-patched line (patches/hook2 first on PYTHONPATH).

Runs the ordinary per-rank cache hook, then -- only when ODDE_TRUNK_PATCH=1 --
arms BOTH direct-window patches (patches/odde_dense_trunk_patch.py) through the
kit's own opt_core.autoload.patch_attr_at_import: one AttrPatch per site, both on
opendde.model.modules.primitives, which AttrPatch fires together at that one
import. Nothing heavy is imported here.
"""
import os
import runpy
import sys

_ROOT = os.environ.get("ODDE_ROOT", "/scratch/pawsey1018/gbouras/opendde-setonix")
_BASE = os.path.join(_ROOT, "scripts", "hooks", "sitecustomize.py")
if os.path.isfile(_BASE):
    runpy.run_path(_BASE)

if os.environ.get("ODDE_TRUNK_PATCH") == "1":
    _PATCHES = os.path.join(_ROOT, "patches")
    if _PATCHES not in sys.path:
        sys.path.insert(0, _PATCHES)
    try:
        import odde_dense_trunk_patch as _p
        from opt_core import autoload as _al
        _T = "opendde.model.modules.primitives"
        _al.patch_attr_at_import(_T, "rearrange_qk_to_dense_trunk", _p.make_patched_qk_logged,
                                 tag="odde-trunk-patch", name="dense_mask")
        _al.patch_attr_at_import(_T, "rearrange_to_dense_trunk", _p.make_patched_dense_logged,
                                 tag="odde-trunk-patch", name="dense_bias")
    except Exception as _e:
        sys.stderr.write("[odde-trunk-patch] NOT INSTALLED: %s: %s\n" % (type(_e).__name__, _e))
        sys.stderr.flush()
        os._exit(3)
