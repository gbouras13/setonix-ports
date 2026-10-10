#!/usr/bin/env python3
"""colabfold_check.py [--no-gpu] -- the ColabFold image's self-check (run as `colabfold check`). Exit 0 only when every check passes.

--no-gpu (the image build runs this):
  - the installed versions equal the port's lock (/opt/colabfold/environment/requirements.lock, the native venv's freeze);
  - the kit's own pin check (kitpin/stock/check_pins.py --checks packages,files) passes;
  - the kit packages, the boot patch and the nvidia-smi shim are in place;
  - ColabFold and AlphaFold import;
  - every shared object of XLA's ROCm plugin, and every ROCm library XLA opens at run time, resolves inside the image (ldd);
  - the GPU MMseqs2 runs and was built with its GPU options.
Default (a GPU node): all of that, plus:
  - the devices JAX sees (backend gpu, every one gfx90a), and a bf16 matmul on each;
  - where every GPU library mapped into this process came from: all must be the image's own (/opt/rocm-<version>/...);
  - the nvidia-smi shim lists the same GPUs;
  - an MMseqs2 search on the GPU of a few sequences against a padded copy of themselves finds every self-hit.
"""
import glob
import importlib.metadata as md
import importlib.util
import os
import re
import subprocess
import sys
import tempfile

W = "/opt/colabfold"
LOCK = f"{W}/environment/requirements.lock"
MMSEQS = f"{W}/bin/mmseqs"
GPU_LIB = re.compile(r"/lib(amdhip64|hsa-runtime64|hsakmt|amd_comgr|hipblaslt|rocblas|hipblas|MIOpen|rccl|roctracer64|rocprofiler[-_a-z]*|"
                     r"rocm_smi64|hipsparse|rocsparse|hipfft|rocfft|hipsolver|rocsolver|hiprand|rocrand|drm_amdgpu)\.so[.0-9]*$")
RUNTIME_LIBS = ("amdhip64", "hsa-runtime64", "amd_comgr", "hipblaslt", "rocblas", "hipblas", "MIOpen", "rccl", "roctracer64", "rocsolver",
                "hipsolver", "rocfft", "hipfft", "rocsparse", "hipsparse", "rocrand", "hiprand", "rocm_smi64")
FAILS = []


def say(tag, msg):
    print(f"[check] {tag:<9} {msg}", flush=True)


def fail(msg):
    FAILS.append(msg)
    say("FAIL", msg)


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def check_versions():
    want = {}
    for line in open(LOCK):
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)", line.strip())
        if m:
            want[norm(m.group(1))] = m.group(2)
    have = {norm(d.metadata["Name"]): d.version for d in md.distributions()}
    bad = [f"{n}: lock {v}, installed {have.get(n, 'ABSENT')}" for n, v in sorted(want.items()) if have.get(n) != v]
    for b in bad:
        fail(f"version {b}")
    key = ("colabfold", "alphafold-colabfold", "jax", "jaxlib", "jax-rocm7-plugin", "jax-rocm7-pjrt", "dm-haiku", "numpy")
    say("versions", f"{len(want) - len(bad)}/{len(want)} lock pins installed exactly; " + " ".join(f"{k}={have.get(k, '-')}" for k in key))


def check_kit():
    for mod in ("colabfold_opt", "opt_core"):
        spec = importlib.util.find_spec(mod)
        if spec is None:
            fail(f"kit package {mod} is not installed")
        else:
            say("kit", f"{mod} at {os.path.dirname(spec.origin or '') or list(spec.submodule_search_locations or [])}")
    r = subprocess.run([sys.executable, "-I", f"{W}/kitpin/stock/check_pins.py", "--checks", "packages,files"], capture_output=True, text=True)
    for line in (r.stdout + r.stderr).strip().splitlines()[-6:]:
        say("pins", line[:200])
    if r.returncode != 0:
        fail(f"kitpin/stock/check_pins.py --checks packages,files: rc {r.returncode}")
    for f in (f"{W}/kitpin/run.sh", f"{W}/kitpin/configs/mi250x.env", f"{W}/scripts/boot/sitecustomize.py", f"{W}/scripts/cfs-run.sh",
              f"{W}/scripts/cf-search.sh"):
        if not os.path.isfile(f):
            fail(f"missing {f}")
    if not os.access(f"{W}/scripts/bin/nvidia-smi", os.X_OK):
        fail(f"{W}/scripts/bin/nvidia-smi is not executable")
    with open(f"{W}/kitpin/configs/mi250x.env") as f:
        jit = [l.strip() for l in f if "COLABFOLD_OPT_JIT_ROOT:-" in l]
    say("kit", f"mi250x.env compile-cache default: {jit[0][:120] if jit else '?'}")


def check_colabfold():
    try:
        import colabfold.batch  # noqa: F401  (imports jax and alphafold)
        from alphafold.model import model  # noqa: F401
    except Exception as e:  # noqa: BLE001
        fail(f"colabfold import: {type(e).__name__}: {e}")
        return
    say("colabfold", f"colabfold {md.version('colabfold')} and alphafold-colabfold {md.version('alphafold-colabfold')} import")


def ldd(sos):
    """{missing library: {objects needing it}}, and the directories the plugin's ROCm libraries resolve from"""
    missing, dirs = {}, set()
    for so in sos:
        out = subprocess.run(["ldd", so], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if "not found" in line:
                missing.setdefault(line.strip().split()[0], set()).add(os.path.basename(so))
            m = re.search(r"=> (/\S+/)lib(amdhip64|hipsparse|MIOpen|hipblas|rocblas|rccl)\S*\.so", line)
            if m:
                dirs.add(m.group(1))
    return missing, dirs


def check_plugin_libs():
    sos = []
    for mod in ("jax_plugins.xla_rocm7", "jax_rocm7_plugin"):
        spec = importlib.util.find_spec(mod)
        if spec is None or not spec.submodule_search_locations:
            fail(f"{mod} is not installed")
            continue
        for d in spec.submodule_search_locations:
            sos += glob.glob(os.path.join(d, "**", "*.so*"), recursive=True)
    missing, dirs = ldd(sos)
    for lib, users in sorted(missing.items()):
        fail(f"ROCm plugin needs {lib}: not found in the image (via {', '.join(sorted(users))})")
    say("plugin", f"{len(sos)} shared objects; ROCm libraries resolve from {sorted(dirs) or '-'}; unresolved: {len(missing)}")


def check_runtime_libs():
    lib = os.path.realpath("/opt/rocm/lib")
    sos = []
    for name in RUNTIME_LIBS:
        hits = glob.glob(os.path.join(lib, f"lib{name}.so.*"))
        if not hits:
            fail(f"lib{name}.so is not installed in {lib}")
            continue
        sos.append(max(hits, key=len))   # the fully versioned file
    missing, _ = ldd(sos)
    for name, users in sorted(missing.items()):
        fail(f"{name} not found in the image (needed by {', '.join(sorted(users))})")
    say("rocm", f"{len(sos)} run-time ROCm libraries in {lib}; unresolved: {len(missing)}")


def check_mmseqs():
    if not os.access(MMSEQS, os.X_OK):
        fail(f"no {MMSEQS}")
        return
    r = subprocess.run([MMSEQS, "version"], capture_output=True, text=True)
    if r.returncode != 0:
        fail(f"mmseqs version: rc {r.returncode}: {r.stderr.strip()[:200]}")
        return
    missing, _ = ldd([MMSEQS])
    for name, _ in sorted(missing.items()):
        fail(f"mmseqs needs {name}: not found in the image")
    h = subprocess.run([MMSEQS, "search", "-h"], capture_output=True, text=True)
    if "--gpu-server" not in h.stdout + h.stderr:
        fail("mmseqs was built without its GPU options (no --gpu-server)")
    build = open(f"{W}/bin/mmseqs-build.txt").read().strip().splitlines()[0] if os.path.isfile(f"{W}/bin/mmseqs-build.txt") else "?"
    say("mmseqs", f"{r.stdout.strip()} ({build}); unresolved libraries: {len(missing)}")


def mapped_gpu_libs():
    libs = {}
    for line in open("/proc/self/maps"):
        f = line.split()
        if len(f) >= 6 and GPU_LIB.search(f[-1]):
            libs[os.path.basename(f[-1])] = os.path.realpath(f[-1])
    return libs


def report_libs(stage, libs, seen):
    """Print the GPU libraries mapped since the last report, and fail any that is not the image's own."""
    real = os.path.realpath("/opt/rocm")
    new = {k: v for k, v in libs.items() if k not in seen}
    say("libs", f"{stage}: {len(new)} more GPU libraries mapped (image ROCm at {real})")
    for k, v in sorted(new.items()):
        say("", f"{k:<34} {v}")
        if not (v.startswith(real + "/") or v.startswith("/opt/amdgpu/") or v.startswith("/usr/lib/")):
            fail(f"{k} loaded from OUTSIDE the image: {v} (run with --rocm and without the entry point, or a forced LD_LIBRARY_PATH?)")
    seen.update(new)


def check_gpu():
    import jax
    import jax.numpy as jnp
    ds = jax.devices()                     # backend init maps the HIP / HSA runtime: report their origin BEFORE any kernel runs
    backend = jax.default_backend()
    ccs = sorted({str(getattr(d, "compute_capability", "?")) for d in ds})
    say("devices", f"backend={backend} n={len(ds)} kind={ds[0].device_kind if ds else '-'} cc={ccs} "
                   f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES', '(unset)')}")
    seen = {}
    report_libs("after backend init", mapped_gpu_libs(), seen)
    if backend != "gpu" or not ds or ccs != ["gfx90a"]:
        fail(f"expected gfx90a GPUs on the gpu backend, got backend={backend} cc={ccs}")
        return
    x = jnp.ones((512, 512), jnp.bfloat16)
    sums = [float((jax.device_put(x, d) @ jax.device_put(x, d)).astype(jnp.float32).sum()) for d in ds]
    ok = all(s == 134217728.0 for s in sums)
    (say if ok else fail)("matmul" if ok else "matmul wrong", f"bf16 512^3 on each device: {sorted(set(sums))} (expect 134217728)")
    report_libs("after the first kernels", mapped_gpu_libs(), seen)
    smi = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap", "--format=csv,noheader"], capture_output=True, text=True)
    rows = [l for l in smi.stdout.strip().splitlines() if l.strip()]
    (say if len(rows) == len(ds) else fail)("shim", f"nvidia-smi shim lists {len(rows)} GPU(s) for JAX's {len(ds)}: {rows[0] if rows else smi.stderr.strip()}")
    try:
        say("driver", f"host amdgpu kernel driver {open('/sys/module/amdgpu/version').read().strip()}")
    except OSError:
        pass


def check_mmseqs_gpu():
    """A GPU search of a few sequences against a padded copy of themselves: every query must find itself."""
    seqs = {"a": "MRIETKKSKLSRPGQKAEADLLSDYMVGKEDDPILLNGIDLEHSSKVLIDNKELGI", "b": "MSNEQLNELLTKAAELAKEAGEKAVQLGIEALKNGDKEKAIEL",
            "c": "MKKLLIAGLAVSLLATPAFAQEYKGTVTKVDGDTVTIKTDDGKEMTFKVDAAT"}
    with tempfile.TemporaryDirectory() as d:
        fa = os.path.join(d, "q.fasta")
        with open(fa, "w") as f:
            f.writelines(f">{k}\n{v}\n" for k, v in seqs.items())
        steps = [["createdb", fa, f"{d}/q"], ["makepaddedseqdb", f"{d}/q", f"{d}/t"],
                 ["search", f"{d}/q", f"{d}/t", f"{d}/r", f"{d}/tmp", "--gpu", "1", "--prefilter-mode", "1", "--threads", "4"],
                 ["convertalis", f"{d}/q", f"{d}/t", f"{d}/r", f"{d}/r.m8"]]
        for s in steps:
            r = subprocess.run([MMSEQS] + s, capture_output=True, text=True)
            if r.returncode != 0:
                fail(f"mmseqs {s[0]} on the GPU: rc {r.returncode}: {(r.stderr or r.stdout).strip().splitlines()[-1:]}")
                return
        hits = {tuple(l.split("\t")[:2]) for l in open(f"{d}/r.m8")}
        selfs = sum((k, k) in hits for k in seqs)
        (say if selfs == len(seqs) else fail)("mmseqs", f"GPU search: {selfs}/{len(seqs)} self-hits")


def main():
    no_gpu = "--no-gpu" in sys.argv[1:]
    if no_gpu:
        os.environ["JAX_PLATFORMS"] = "cpu"   # no GPU here (the image build): any import that touches a backend gets the CPU, not an error
    for f in (f"{W}/source.txt", f"{W}/kit.txt"):
        if os.path.isfile(f):
            say("image", open(f).read().strip())
    say("python", f"{sys.version.split()[0]} at {sys.executable}")
    check_versions()
    check_kit()
    check_colabfold()
    check_plugin_libs()
    check_runtime_libs()
    check_mmseqs()
    if not no_gpu:
        check_gpu()
        check_mmseqs_gpu()
    print(f"[check] {'PASS' if not FAILS else 'FAIL (' + str(len(FAILS)) + ')'}", flush=True)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
