"""sitecustomize for v2 + the chunk floor (patches/hook3 first on PYTHONPATH).

Runs patches/hook2/sitecustomize.py unchanged -- the per-rank cache hook and, with
ODDE_TRUNK_PATCH=1, both byte-identical dense-window replacements -- and then,
only when ODDE_CHUNK_FLOOR=1, arms the chunk floor on
opendde.model.opendde.OpenDDE._resolve_pairformer_chunk_size through the kit's
own opt_core.autoload.patch_attr_at_import. The validated v2 path is untouched:
this is a separate hook directory and a separate flag.
"""
import os
import runpy
import sys

_ROOT = os.environ.get("ODDE_ROOT", "/scratch/pawsey1018/gbouras/opendde-setonix")
_V2 = os.path.join(_ROOT, "patches", "hook2", "sitecustomize.py")
if os.path.isfile(_V2):
    runpy.run_path(_V2)

if os.environ.get("ODDE_CHUNK_FLOOR") == "1":
    _PATCHES = os.path.join(_ROOT, "patches")
    if _PATCHES not in sys.path:
        sys.path.insert(0, _PATCHES)
    try:
        import odde_chunk_floor_patch as _cf
        from opt_core import autoload as _al
        _al.patch_attr_at_import("opendde.model.opendde", "OpenDDE._resolve_pairformer_chunk_size",
                                 _cf.make_chunk_floor_logged, tag="odde-chunk-floor", name="chunk_floor")
    except Exception as _e:
        sys.stderr.write("[odde-chunk-floor] NOT INSTALLED: %s: %s\n" % (type(_e).__name__, _e))
        sys.stderr.flush()
        os._exit(3)
