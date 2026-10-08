#!/usr/bin/env python3
"""af3jax_check.py [--no-gpu] -- the af3_jax image's self-check (run as `af3jax check`). Exit 0 only when every check passes.

--no-gpu (the image build runs this): the installed versions equal the image's ROCm lock (/opt/af3jax/requirements-rocm.lock); both ROCm
patches report APPLIED; AlphaFold 3 imports and its CCD pickle exists (build_data ran); every shared object of XLA's ROCm plugin resolves
all its libraries (ldd, no "not found").
Default (a GPU node): all of that, plus the devices JAX sees (backend gpu, every one gfx90a), a bf16 matmul on each (512^3 of ones = 134217728),
and the check that matters for a container: where every GPU library mapped into this process came from. All must be the image's own
(/opt/rocm-<version>/...); a path under /.singularity.d/ (what `singularity --rocm` binds from the host), /software/ (a host module) or another
ROCm release fails the check.
"""
import glob
import importlib.metadata as md
import importlib.util
import os
import re
import subprocess
import sys

LOCK = "/opt/af3jax/requirements-rocm.lock"
PATCHES = [["/opt/af3jax/scripts/patch_rocm_capability.py", "/opt/af3jax/alphafold", "--check"]]
GPU_LIB = re.compile(r"/lib(amdhip64|hsa-runtime64|hsakmt|amd_comgr|hipblaslt|rocblas|hipblas|MIOpen|rccl|roctracer64|rocprofiler[-_a-z]*|"
                     r"rocm_smi64|hipsparse|rocsparse|hipfft|rocfft|hipsolver|rocsolver|hiprand|rocrand|drm_amdgpu)\.so[.0-9]*$")
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
        elif line.startswith("torch @"):
            want["torch"] = re.search(r"torch-([0-9][^-]*?)-cp", line).group(1).replace("%2B", "+")
    have = {norm(d.metadata["Name"]): d.version for d in md.distributions()}
    bad = [f"{n}: lock {v}, installed {have.get(n, 'ABSENT')}" for n, v in sorted(want.items()) if have.get(n) != v]
    for b in bad:
        fail(f"version {b}")
    key = ("jax", "jaxlib", "jax-rocm7-plugin", "jax-rocm7-pjrt", "dm-haiku", "tokamax", "numpy", "alphafold3-open")
    say("versions", f"{len(want) - len(bad)}/{len(want)} lock pins installed exactly; " + " ".join(f"{k}={have.get(k, '-')}" for k in key))


def check_patches():
    site = [p for p in sys.path if p.endswith("site-packages")][0]
    for cmd in PATCHES + [["/opt/af3jax/scripts/patch_tokamax_rocm.py", site, "--check"]]:
        r = subprocess.run([sys.executable] + cmd, capture_output=True, text=True)
        for line in (r.stdout + r.stderr).strip().splitlines():
            say("patch", line)
        if r.returncode != 0:
            fail(f"patch not applied: {os.path.basename(cmd[0])}")


def check_alphafold():
    try:
        import alphafold3
        from alphafold3.model import model  # noqa: F401  (imports jax, haiku, tokamax)
    except Exception as e:  # noqa: BLE001
        fail(f"alphafold3 import: {type(e).__name__}: {e}")
        return
    pk = os.path.join(os.path.dirname(alphafold3.__file__), "constants", "converters", "ccd.pickle")
    if os.path.isfile(pk):
        say("alphafold", f"alphafold3 at {os.path.dirname(alphafold3.__file__)}; CCD pickle {os.path.getsize(pk) / 2**20:.0f} MiB")
    else:
        fail(f"no {pk}: build_data did not run")


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


# The ROCm libraries XLA opens at run time (dlopen), which ldd on the plugin cannot see: each must be installed and resolve everything
# it links, or the first GPU job fails on Setonix instead of the image build failing here.
RUNTIME_LIBS = ("amdhip64", "hsa-runtime64", "amd_comgr", "hipblaslt", "rocblas", "hipblas", "MIOpen", "rccl", "roctracer64", "rocsolver",
                "hipsolver", "rocfft", "hipfft", "rocsparse", "hipsparse", "rocrand", "hiprand", "rocm_smi64")


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
            fail(f"{k} loaded from OUTSIDE the image: {v} (run with --rocm and without af3jax, or a forced LD_LIBRARY_PATH?)")
    seen.update(new)


def check_gpu():
    import jax
    import jax.numpy as jnp
    ds = jax.devices()                     # backend init maps the HIP / HSA runtime: report their origin BEFORE any kernel runs
    backend = jax.default_backend()
    ccs = sorted({str(getattr(d, "compute_capability", "?")) for d in ds})
    say("devices", f"backend={backend} n={len(ds)} kind={ds[0].device_kind if ds else '-'} cc={ccs} "
                   f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES', '(unset)')}")
    ver = open("/opt/rocm/.info/version").read().strip() if os.path.isfile("/opt/rocm/.info/version") else "?"
    say("rocm", f"image ROCm {ver}; HIP_LAUNCH_BLOCKING={os.environ.get('HIP_LAUNCH_BLOCKING', '(unset)')} "
                f"AMD_SERIALIZE_COPY={os.environ.get('AMD_SERIALIZE_COPY', '(unset)')}")
    seen = {}
    report_libs("after backend init", mapped_gpu_libs(), seen)
    if backend != "gpu" or not ds or ccs != ["gfx90a"]:
        fail(f"expected gfx90a GPUs on the gpu backend, got backend={backend} cc={ccs}")
        return
    x = jnp.ones((512, 512), jnp.bfloat16)
    sums = []
    for d in ds:
        y = jax.device_put(x, d)
        sums.append(float((y @ y).astype(jnp.float32).sum()))
    ok = all(s == 134217728.0 for s in sums)
    (say if ok else fail)("matmul" if ok else "matmul wrong", f"bf16 512^3 on each device: {sorted(set(sums))} (expect 134217728)")
    report_libs("after the first kernels", mapped_gpu_libs(), seen)
    try:
        say("driver", f"host amdgpu kernel driver {open('/sys/module/amdgpu/version').read().strip()}")
    except OSError:
        pass


def main():
    no_gpu = "--no-gpu" in sys.argv[1:]
    for f in ("/opt/af3jax/source.txt", "/opt/af3jax/kit.txt", "/opt/af3jax/wheel.txt"):
        if os.path.isfile(f):
            say("image", open(f).read().strip())
    say("python", f"{sys.version.split()[0]} at {sys.executable}")
    check_versions()
    check_patches()
    check_alphafold()
    check_plugin_libs()
    check_runtime_libs()
    if not no_gpu:
        check_gpu()
    print(f"[check] {'PASS' if not FAILS else 'FAIL (' + str(len(FAILS)) + ')'}", flush=True)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
