"""Start-up fixes applied IN MEMORY, so the kit's own gates and probes still judge the result.

Nothing in the kit tree is edited. This file is imported by `site` at interpreter start-up when its directory is on
`PYTHONPATH` (`scripts/rfd-env.sh` puts it there when `RFD_BOOT` names a fix), which is before anything imports
`rfd3.model`, so a fix can be in place by the time a lever reads its table. The kit's stock child runs under `python -I`
and ignores `PYTHONPATH`, so `--mode off` is never touched by this file — deliberately.

`RFD_BOOT` is a comma-separated list of fix names; unset or empty means this file does nothing at all.

  fzt_rows_off   give the running compute capability an EMPTY served-row range in the fused transition's own table
                 (`rfd3.model.layers.layer_utils.FZT_ROWS_BY_CC`). The table has a row for capability 8.0 only, so at 9.0 —
                 which is what `torch.cuda.get_device_capability()` returns on gfx90a — every row count is fused, on the
                 stated grounds that cuBLAS keeps one summation order there. This card's GEMMs are hipBLASLt. With an empty
                 range the lever steps aside BY NAME, which is a route the kit itself defines ("a stock route by design, not
                 a fallback and not a partial activation"), counted in `cells_stock['rows<MIN:<cell>']`.
"""
import os
import sys

_FIXES = [f.strip() for f in (os.environ.get("RFD_BOOT") or "").split(",") if f.strip()]
if _FIXES:
    _LAYER_UTILS = "rfd3.model.layers.layer_utils"

    def _fzt_rows_off(mod):
        """Every cell of the fused transition gets an impossible row range on this card's capability."""
        import torch
        cc = torch.cuda.get_device_capability(0)
        key = f"{cc[0]}.{cc[1]}"
        cells = getattr(mod, "FZT_CELLS", ())
        mod.FZT_ROWS_BY_CC = dict(getattr(mod, "FZT_ROWS_BY_CC", {}))
        mod.FZT_ROWS_BY_CC[key] = {cell: (1 << 62, 1 << 62) for cell in cells}
        probe = mod._fzt_rows_word(cells[0], 4096, key) if cells and hasattr(mod, "_fzt_rows_word") else "?"
        print(f"[rfd-boot] fzt_rows_off: {_LAYER_UTILS}.FZT_ROWS_BY_CC['{key}'] = no served rows ({len(cells)} cells); "
              f"a 4096-row call of {cells[0] if cells else '?'} now reads {probe!r} - the fused transition steps aside by "
              f"name on this card (the proof is the kit's own APPLIED line: fused_calls=0)", file=sys.stderr)

    _ON_IMPORT = {_LAYER_UTILS: [("fzt_rows_off", _fzt_rows_off)]}

    class _PostImport:
        """Wrap the REAL loader, do not pre-import.

        A finder that imports the module itself and patches the result loses: `_find_and_load` consults finders only
        because the module is not in `sys.modules` yet, and once a spec comes back it loads a FRESH module object over
        whatever the finder put there — so the patch lands on an object that is then discarded (seen once: the fix printed
        its line and the lever still reported `fused_calls=56`). Wrapping `exec_module` on the real spec patches the module
        the import system actually keeps.
        """

        def find_spec(self, fullname, path=None, target=None):
            if fullname not in _ON_IMPORT:
                return None
            import importlib.util
            sys.meta_path.remove(self)                      # let the normal finders answer, without recursing into this one
            try:
                spec = importlib.util.find_spec(fullname)
            except Exception:                               # noqa: BLE001 - an import that was going to fail anyway
                spec = None
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
                return None
            _real = spec.loader.exec_module

            def exec_module(module, _real=_real, _name=fullname):
                _real(module)
                for name, fn in _ON_IMPORT[_name]:
                    if name in _FIXES:
                        try:
                            fn(module)
                        except Exception as e:  # noqa: BLE001 - a fix that cannot apply says so and the run goes on
                            print(f"[rfd-boot] {name}: NOT applied ({type(e).__name__}: {e})", file=sys.stderr)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _PostImport())
    print(f"[rfd-boot] armed: {','.join(_FIXES)}", file=sys.stderr)
