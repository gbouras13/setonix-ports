"""sitecustomize.py -- the gfx90a boot patch of the ColabFold kit port (Setonix MI250X). On PYTHONPATH only for kit-mode runs
(scripts/cfs-run.sh puts scripts/boot there); inert unless CFS_GFX90A=1 AND COLABFOLD_OPT names a kit mode (exact | fast | big), so
`--mode off` (whose environment the kit strips of COLABFOLD_OPT*) and every other process run untouched. Edits no kit or core file.

What it does: jax's ROCm plugin reports a device's compute capability as the string 'gfx90a'. The shared core's JAX provider
(opt_core.kernels.pallas) parses a capability with cc_of(), which accepts only NVIDIA forms ('9.0', (9, 0), 'sm_90') and raises
ValueError on 'gfx90a' -- not the provider's Refusal, so the kit's levers (which pass jax's string, e.g. trimul_pallas._selection)
would crash at trace time instead of stepping aside. The patch wraps cc_of so that a 'gfx...' string reads as $OPT_CORE_PALLAS_CC
(the provider's own off-target selection variable; the port sets 8.0 = the A100's measured column). Every Pallas row those cells name
was run on gfx90a against the stock statement (tools/probe_rows.py, smoke 2: attention rows rel-RMS ~8e-3, transition ~5e-5, trimul
~6e-3); rows that check the device themselves refuse BY NAME (fpf_*: no_tiles:gfx90a; tokamax: not installed) and the provider's walk
serves the next one. An unmeasured word (8.1, smoke 1) would route every row through the provider's first-use probe, which fails on
jax 0.10 under jit (NotImplementedError) whatever the kernel. Everything else (the kit's census lines keep cc=gfx90a) is unchanged.
"""
import os
import sys

_KIT_MODES = ("exact", "fast", "big")


def _active():
    return (os.environ.get("CFS_GFX90A") == "1"
            and (os.environ.get("COLABFOLD_OPT") or "").strip().lower() in _KIT_MODES)


def _say(msg):
    try:
        sys.stderr.write("[cfs-gfx90a] %s\n" % msg)
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass


def _patch_pallas(mod):
    orig = getattr(mod, "cc_of", None)
    if orig is None or getattr(orig, "_cfs_gfx90a", False):
        return
    synthetic = os.environ.get("OPT_CORE_PALLAS_CC") or "8.0"

    def cc_of(cc):
        if isinstance(cc, str) and cc.strip().lower().startswith("gfx"):
            return orig(synthetic)
        return orig(cc)

    cc_of._cfs_gfx90a = True
    cc_of.__wrapped__ = orig
    cc_of.__doc__ = orig.__doc__
    mod.cc_of = cc_of
    _say("pid %d: %s.cc_of reads 'gfx*' as %s (OPT_CORE_PALLAS_CC: the provider column gfx90a is served from)"
         % (os.getpid(), mod.__name__, synthetic))


# big --n_gpu P > 1 (approved by the user, 2026-10-05): the shared core's AlphaFold-2 row-shard recipe (opt_core.mem.rowpair_jax.alphafold)
# refuses by name any modules.py / modules_multimer.py whose sha256 is not in its PIN table (the alphafold-colabfold 2.3.13 files; folding.py,
# folding_multimer.py and confidence.py are byte-identical in 2.3.20 and stay pinned as they are). The recipe's own rule: "a kit on another
# tree passes pin= only together with its own review". Review (COLABFOLD_PORT.md, "The sharding recipe's pin"): under bfloat16=True and
# use_pallas=False (stock 1.6.3's default; the kit never sets use_pallas) the 2.3.20 hunks in the rebound classes are fp16-only helpers
# (wide_einsum / to_half_like: identities for bf16), Pallas-only branches (fused_ops.* return None), constant-equal sentinels (mask_to_bias
# = -1e9 rounded to the activation dtype, as 1e9*(mask-1) is on a bf16 mask; logit_clip = 1e8), the jnp.clip keyword change and the
# half_context rename. The entry adds the two 2.3.20 shas with a 2.3.13-tree label (the recipe branches on that tree word: PIN_TREE,
# CONFIDENCE_IN_GRAPH). Nothing else is changed; CFS_ROWPAIR_PIN=off leaves the table as shipped (P > 1 then refuses again).
_PIN_2320 = {
    "modules": ("1d4f02c3f2d8a25b246b82da1d82a1698a08d28bf041c006a9142d78108dbbc1", "alphafold/model/modules.py"),
    "modules_multimer": ("f106384325307440aed91b927461c123941fafd1d99368555471209dfbe1caea", "alphafold/model/modules_multimer.py"),
}


def _patch_rowpair_pin(mod):
    if (os.environ.get("CFS_ROWPAIR_PIN") or "2.3.20") == "off" or getattr(mod, "_cfs_pin_2320", False):
        return
    for kind, (sha, rel) in _PIN_2320.items():
        label = ("%s %s statements as shipped in alphafold-colabfold 2.3.20 (bytes differ: fp16 helpers, use_pallas branches; "
                 "reviewed for the Setonix port, 2026-10-05)" % (mod.TREE_CF, rel))
        mod.PIN_FILES[kind][sha] = label
        mod.PIN[sha] = label
        mod.PIN_TREE[sha] = mod.TREE_CF
    mod._cfs_pin_2320 = True
    _say("pid %d: %s PIN += alphafold-colabfold 2.3.20 modules.py %s, modules_multimer.py %s (reviewed; CFS_ROWPAIR_PIN=off refuses)"
         % (os.getpid(), mod.__name__, _PIN_2320["modules"][0][:12], _PIN_2320["modules_multimer"][0][:12]))


_TARGETS = {"opt_core.kernels.pallas": _patch_pallas, "opt_core.mem.rowpair_jax.alphafold": _patch_rowpair_pin}


def _install():
    import importlib.abc

    class _PostImport(importlib.abc.MetaPathFinder):
        """Find a target module through the other finders, then run the patch right after the module body executes (before any module
        that does `from opt_core.kernels.pallas import cc_of` binds the name)."""

        def find_spec(self, fullname, path, target=None):
            if fullname not in _TARGETS:
                return None
            spec = None
            for f in sys.meta_path:
                if f is self or not hasattr(f, "find_spec"):
                    continue
                spec = f.find_spec(fullname, path, target)
                if spec is not None:
                    break
            if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
                return spec
            loader, patch = spec.loader, _TARGETS[fullname]
            inner = loader.exec_module

            def exec_module(module):
                inner(module)
                patch(module)

            loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _PostImport())
    for name, patch in _TARGETS.items():          # already imported (not expected at interpreter start): patch in place
        if name in sys.modules:
            patch(sys.modules[name])


if _active():
    os.environ.setdefault("OPT_CORE_PALLAS_CC", "8.0")
    _install()
