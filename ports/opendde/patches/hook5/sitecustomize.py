"""sitecustomize for v2 + chunk floor + explicit blocks + the native allocator (patches/hook5 first on PYTHONPATH).

Runs patches/hook4/sitecustomize.py unchanged (hook3 -> hook2 -> the base per-rank hook; v2 with ODDE_TRUNK_PATCH=1,
the floor with ODDE_CHUNK_FLOOR=1, the blocks with ODDE_TP_BLOCKS=1). Then, only when ODDE_NATIVE_ALLOC=1, it arms
patches/odde_native_alloc_patch.py on opt_core.mem.torch_alloc.conf_for and opendde_opt.modes.resolve through the kit's own
opt_core.autoload.patch_attr_at_import. The validated paths are untouched: this is a separate hook directory and a
separate flag. With ODDE_ABLOCK_WHOLE_FRAC set it also arms patches/odde_ablock_patch.py on
opt_core.mem.rowpair.trimul.ablock_auto_cap. Fails closed (exit 3) if either cannot be armed.
"""
import os
import runpy
import sys

_ROOT = os.environ.get("ODDE_ROOT", "/scratch/pawsey1018/gbouras/opendde-setonix")
_H4 = os.path.join(_ROOT, "patches", "hook4", "sitecustomize.py")
if os.path.isfile(_H4):
    runpy.run_path(_H4)

if os.environ.get("ODDE_NATIVE_ALLOC") == "1":
    _PATCHES = os.path.join(_ROOT, "patches")
    if _PATCHES not in sys.path:
        sys.path.insert(0, _PATCHES)
    try:
        import odde_native_alloc_patch as _na
        from opt_core import autoload as _al
        _al.patch_attr_at_import("opt_core.mem.torch_alloc", "conf_for", _na.make_conf_for_logged,
                                 tag="odde-native-alloc", name="conf_for")
        _al.patch_attr_at_import("opendde_opt.modes", "resolve", _na.make_resolve_logged,
                                 tag="odde-native-alloc", name="resolve")
    except Exception as _e:
        sys.stderr.write("[odde-native-alloc] NOT INSTALLED: %s: %s\n" % (type(_e).__name__, _e))
        sys.stderr.flush()
        os._exit(3)

if os.environ.get("ODDE_ABLOCK_WHOLE_FRAC"):
    _PATCHES = os.path.join(_ROOT, "patches")
    if _PATCHES not in sys.path:
        sys.path.insert(0, _PATCHES)
    try:
        import odde_ablock_patch as _ab
        from opt_core import autoload as _al
        _ab.frac()                        # refuse a bad value before anything runs
        _al.patch_attr_at_import("opt_core.mem.rowpair.trimul", "ablock_auto_cap", _ab.make_auto_cap_logged,
                                 tag="odde-ablock", name="ablock_auto_cap")
    except Exception as _e:
        sys.stderr.write("[odde-ablock] NOT INSTALLED: %s: %s\n" % (type(_e).__name__, _e))
        sys.stderr.flush()
        os._exit(3)
