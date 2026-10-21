"""Start-up fixes for the ESMFold2 kit on Setonix MI250X, applied in memory per RUN, never to the kit tree or upstream's files.

scripts/ef2-env.sh puts this directory on PYTHONPATH; `site` imports this file at interpreter start. EFS_BOOT names the fixes for one run
(comma list); unset or empty = this file does nothing at all. Each fix hooks the module it changes by wrapping that module's real loader
(a meta-path finder that never pre-imports the module), and prints one [efs-boot] line when it is applied.

  dotprec  Triton's AMD backend accepts tl.dot input_precision 'ieee', 'bf16x3', 'bf16x6' only (gfx90a; 'tf32' on gfx942 alone).
           Upstream's fused backend asks for 'tf32x3' in its pair-bias kernel (fused_attention_pair_bias.py:146), so `--backend fused`
           -- the configuration `exact` reproduces -- cannot compile here (smoke2 50989775: CompilationError; smoke1 50947550 died with
           SIGSEGV at the same point). The kit's dit / af levers and shared-core members ask for 'tf32' / 'tf32x3' too. 'tf32x3' means
           "fp32-faithful products on tensor cores"; MI250X has native fp32 matrix instructions, so the faithful translation is 'ieee'
           (true fp32 products, fp32 accumulation). This fix maps 'tf32' / 'tf32x3' to 'ieee' when (and only when) the backend's
           allowed list lacks them; every other value is passed through unchanged. Applies to the stock route too: the kernel is upstream's.
  smem     torch's ROCm build has no `shared_memory_per_block_optin` on its device properties (it reads None here). The kit and the shared
           core size their Triton tiles from that field, and where it is missing they assume an H100 from compute capability 9.0, which
           is what gfx90a reports: 227 KB (ef2_w4._device_info, opt_core ln / transition / nvjit tables). This card has 64 KB of LDS per
           workgroup and no opt-in carve-out beyond it, so the field is made to read the device's own `shared_memory_per_block` (65,536).
"""
import os
import sys

_WORDS = {w.strip() for w in os.environ.get("EFS_BOOT", "").split(",") if w.strip()}
_KNOWN = {"smem", "dotprec"}


def _log(msg):
    sys.stderr.write(f"[efs-boot] {msg}\n")
    sys.stderr.flush()


# ---------------------------------------------------------------------------------------------------------------- smem
class _Props:
    """A device-properties view: every attribute from torch's object, plus shared_memory_per_block_optin when torch leaves it unset."""
    __slots__ = ("_p", "_optin")

    def __init__(self, p):
        object.__setattr__(self, "_p", p)
        v = getattr(p, "shared_memory_per_block_optin", None)
        object.__setattr__(self, "_optin", int(v) if v else int(p.shared_memory_per_block))

    def __getattr__(self, k):
        if k == "shared_memory_per_block_optin":
            return self._optin
        return getattr(self._p, k)

    def __repr__(self):
        return repr(self._p)[:-1] + f", shared_memory_per_block_optin={self._optin} [efs-boot])"


def _patch_torch_cuda(mod):
    orig = mod.get_device_properties
    if getattr(orig, "_efs_smem", False):
        return

    def get_device_properties(device=None):
        return _Props(orig(device))

    get_device_properties._efs_smem = True
    get_device_properties.__doc__ = orig.__doc__
    mod.get_device_properties = get_device_properties
    _log("smem: torch.cuda.get_device_properties(...).shared_memory_per_block_optin now reads the device's shared_memory_per_block")


# ---------------------------------------------------------------------------------------------------------------- dotprec
_DOTPREC_SEEN = set()


def _patch_triton_semantic(mod):
    cls = getattr(mod, "TritonSemantic", None)
    if cls is None:
        _log("dotprec: triton.language.semantic has no TritonSemantic -- NOT applied")
        return
    orig = cls._str_to_dot_input_precision
    if getattr(orig, "_efs_dotprec", False):
        return

    def _str_to_dot_input_precision(self, input_precision):
        p = str(input_precision).lower()
        allowed = tuple(self.builder.options.allowed_dot_input_precisions)
        if p in ("tf32", "tf32x3") and p not in allowed and "ieee" in allowed:
            if p not in _DOTPREC_SEEN:
                _DOTPREC_SEEN.add(p)
                _log(f"dotprec: tl.dot input_precision '{p}' -> 'ieee' (this backend allows {allowed})")
            input_precision = "ieee"
        return orig(self, input_precision)

    _str_to_dot_input_precision._efs_dotprec = True
    cls._str_to_dot_input_precision = _str_to_dot_input_precision
    _log("dotprec: armed in triton.language.semantic.TritonSemantic")


# ---------------------------------------------------------------------------------------------------------------- the hook
_PATCHES = {}
if "smem" in _WORDS:
    _PATCHES["torch.cuda"] = _patch_torch_cuda
if "dotprec" in _WORDS:
    _PATCHES["triton.language.semantic"] = _patch_triton_semantic


class _Finder:
    """Wraps the real spec's exec_module for the hooked modules (patch after the module has run; never pre-import it)."""

    def find_spec(self, name, path=None, target=None):
        fn = _PATCHES.get(name)
        if fn is None:
            return None
        for f in sys.meta_path:
            if f is self or not hasattr(f, "find_spec"):
                continue
            spec = f.find_spec(name, path, target)
            if spec is not None and spec.loader is not None:
                real = spec.loader.exec_module

                def exec_module(module, _real=real, _fn=fn):
                    _real(module)
                    _fn(module)

                spec.loader.exec_module = exec_module
                return spec
        return None


if _WORDS:
    unknown = _WORDS - _KNOWN
    if unknown:
        _log(f"unknown EFS_BOOT word(s) {sorted(unknown)} (known: {sorted(_KNOWN)}); nothing armed")
    else:
        pending = {}
        for name, fn in _PATCHES.items():
            if name in sys.modules:
                fn(sys.modules[name])
            else:
                pending[name] = fn
        _PATCHES = pending
        if _PATCHES:
            sys.meta_path.insert(0, _Finder())
        _log(f"armed: {sorted(_WORDS)}")
