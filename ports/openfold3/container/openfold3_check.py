#!/usr/bin/env python
"""openfold3_check.py -- is this image whole, are its sources the native port's, and is it running on its own libraries?

    openfold3 check [--no-gpu] [--weights DIR] [--gcds 0,1,...]

Sections, each PASS or FAIL, and a verdict line at the end:
  tree        the interpreter, torch (a ROCm build), numpy and rdkit; openfold3 0.5.0, opt_core 0.5.228.0 and openfold3_opt 0.8.20.1 imported
              from the image; the kit's autoload guard in site-packages; the launcher, its environment script and the two runner yamls;
              the tp line's hook directory; the patch markers the launcher looks for; run_openfold on PATH; the C compiler and Python
              headers Triton builds its kernel launchers with; a writable /tmp behind $W/tmp (the ranks' rendezvous file)
  sources     every file of the two kit trees and the three patched OpenFold3 files against environment/native-sources.sha256, the native
              port's own digests: the build checked them once, this checks the image as it runs
  weights     (with weights) the OpenBind checkpoint against the S3 object's size and CRC64NVME (tools/of3_weights.py check)
  devices     the GCDs torch sees, their architecture, and a matmul on each against a CPU reference
  libraries   every loaded GPU library comes from inside the image. Singularity's --rocm binds the HOST's ROCm 6.3.0 over the image's
              7.2 (same sonames) and that is a silent wrong answer, so it is a failure here, not a warning

Exit 0 when every section selected passes, 1 otherwise. Reads nothing outside the image but the weights.
"""
import argparse
import hashlib
import importlib.metadata as md
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import time

W = os.environ.get("OF3_ROOT", "/opt/openfold3")
RESULTS = []
WANT = {"openfold3": "0.5.0", "opt_core": "0.5.228.0", "openfold3_opt": "0.8.20.1"}
MARKERS = {   # the launcher greps the same three; the rest are the patches' own markers in the files they change
    "opt_core_src/opt_core/opt_core/mem/rowpair/trimul.py": "[trimul_hipblaslt]",
    "opt_core_src/opt_core/opt_core/mem/rowpair/pairstack.py": "[transpose_peers]",
    "of3_opt_src/opt/openfold3_opt/tp_rowpair/model.py": "[ln_kernel]",
}


def say(*a):
    print(*a, flush=True)


def section(name, ok, *lines):
    RESULTS.append((name, bool(ok)))
    say("\n== %s: %s" % (name, "PASS" if ok else "FAIL"))
    for ln in lines:
        say("   " + str(ln))
    return ok


def read(path, default=""):
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return default


def inside(path):
    return bool(path) and os.path.realpath(path).startswith(os.path.realpath(W) + os.sep)


# ---------------------------------------------------------------------------------------------------------------- tree
def check_tree():
    import numpy
    lines = ["python %s (%s, environment %s)" % (sys.version.split()[0], sys.executable, sys.prefix), "numpy  %s" % numpy.__version__,
             read(os.path.join(W, "source.txt"), "source unknown"), read(os.path.join(W, "openfold3.txt"), "openfold3 unknown"),
             read(os.path.join(W, "kit.txt"), "kit unknown")]
    ok = os.path.realpath(sys.prefix) == os.path.realpath(os.path.join(W, "py312"))   # the venv's python links to /opt/python's
    if not ok:
        lines.append("NOT the image's environment: sys.prefix %s, want %s" % (sys.prefix, os.path.join(W, "py312")))
    try:
        import torch
        lines.append("torch  %s (hip %s, cuda %s)" % (torch.__version__, torch.version.hip, torch.version.cuda))
        if not torch.version.hip:
            lines.append("NOT a ROCm build of torch")
            ok = False
    except Exception as e:                                                  # noqa: BLE001
        lines.append("torch does not import: %r" % (e,))
        ok = False
    try:
        lines.append("rdkit  %s" % md.version("rdkit"))
    except md.PackageNotFoundError:
        lines.append("rdkit is NOT installed")
        ok = False
    for dist, want in WANT.items():
        try:
            v = md.version(dist)
        except md.PackageNotFoundError:
            v = None
        spec = importlib.util.find_spec(dist)
        origin = getattr(spec, "origin", None)
        good = v == want and inside(origin)
        lines.append("%-14s %-10s %s%s" % (dist, v or "ABSENT", origin or "ABSENT", "" if good else "  <- want %s, inside %s" % (want, W)))
        ok = ok and good
    purelib = sysconfig.get_paths()["purelib"]
    if not os.path.isfile(os.path.join(purelib, "openfold3_opt_autoload.pth")):
        lines.append("MISSING openfold3_opt_autoload.pth in %s (the kit's import guard, as natively)" % purelib)
        ok = False
    for f in ("kit/scripts/predict-sharded.sh", "kit/scripts/setonix-env.sh", "kit/configs/setonix-mi250x-sharded-lazytempl.yml",
              "kit/configs/setonix-mi250x-sharded-lazytempl-c256.yml", "of3_opt_src/opt/openfold3_opt/tp_rowpair/hook",
              "tools/of3_weights.py", "tools/of3_outputs.py", "tools/of3_query.py", "environment/native-sources.sha256"):
        if not os.path.exists(os.path.join(W, f)):
            lines.append("MISSING %s" % f)
            ok = False
    for f, mark in MARKERS.items():
        if mark not in read(os.path.join(W, f)):
            lines.append("patch marker %s absent from %s" % (mark, f))
            ok = False
    ro = shutil.which("run_openfold")
    lines.append("run_openfold %s" % (ro or "ABSENT"))
    ok = ok and inside(ro)
    cc = shutil.which(os.environ.get("CC") or "gcc") or shutil.which("cc")
    header = os.path.join(sysconfig.get_paths()["include"], "Python.h")
    lines.append("C compiler %s, Python.h %s (Triton builds its kernel launchers with them)"
                 % (cc or "ABSENT", "present" if os.path.isfile(header) else "ABSENT"))
    ok = ok and bool(cc) and os.path.isfile(header)
    tmp = os.path.join(W, "tmp")
    probe = os.path.join(tmp, ".of3-probe-%d" % os.getpid())
    try:
        open(probe, "w").close()
        os.remove(probe)
        lines.append("%s -> %s, writable (the ranks' rendezvous file)" % (tmp, os.path.realpath(tmp)))
    except OSError as e:
        lines.append("%s is not writable: %s" % (tmp, e))
        ok = False
    return section("tree", ok, *lines)


# ---------------------------------------------------------------------------------------------------------------- sources
def check_sources():
    manifest = os.path.join(W, "environment", "native-sources.sha256")
    bad, n = [], 0
    for line in read(manifest).splitlines():
        want, rel = line.split(None, 1)
        n += 1
        p = os.path.join(W, rel)
        h = hashlib.sha256()
        try:
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
        except OSError:
            bad.append("MISSING " + rel)
            continue
        if h.hexdigest() != want:
            bad.append("DIFFERS " + rel)
    lines = ["%d files against the native port's digests: %s" % (n, "all identical" if not bad else "%d not" % len(bad))] + bad[:10]
    return section("sources", n > 0 and not bad, *lines)


# ---------------------------------------------------------------------------------------------------------------- weights
def check_weights(root):
    t0 = time.time()
    p = subprocess.run([sys.executable, "-I", os.path.join(W, "tools", "of3_weights.py"), "check", root],
                       capture_output=True, text=True, timeout=1800)
    tail = [ln for ln in (p.stdout + p.stderr).splitlines() if ln.strip()][-3:]
    return section("weights", p.returncode == 0, "weights dir %s (%.0f s)" % (root, time.time() - t0), *tail)


# ---------------------------------------------------------------------------------------------------------------- devices
def check_devices(gcds):
    import torch
    if not torch.cuda.is_available():
        return section("devices", False, "torch.cuda.is_available() is False (no /dev/kfd, or no GPU in this job)")
    n = torch.cuda.device_count()
    want = [int(g) for g in gcds.split(",") if g.strip() != ""] if gcds else list(range(n))
    ok, lines = True, ["%d GCD(s) visible (ROCR_VISIBLE_DEVICES=%s)" % (n, os.environ.get("ROCR_VISIBLE_DEVICES", "unset"))]
    for i in want:
        if i >= n:
            lines.append("GCD %d: not visible" % i)
            ok = False
            continue
        p = torch.cuda.get_device_properties(i)
        a = torch.randn(512, 512, device="cuda:%d" % i, dtype=torch.float32)
        b = torch.randn(512, 512, device="cuda:%d" % i, dtype=torch.float32)
        d = float((a @ b - (a.cpu() @ b.cpu()).to("cuda:%d" % i)).abs().max())
        good = d < 1e-2 and p.gcnArchName.startswith("gfx90a")
        ok = ok and good
        lines.append("GCD %d  %-22s %-22s %5.1f GiB  matmul max|d| %.2e %s"
                     % (i, p.name, p.gcnArchName, p.total_memory / 2**30, d, "" if good else "<- WRONG"))
    return section("devices", ok, *lines)


# ---------------------------------------------------------------------------------------------------------------- libraries
GPU_LIB = re.compile(r"lib(amdhip|hsa-runtime|rocblas|hipblaslt|MIOpen|rocrand|hiprand|rccl|amd_comgr|rocsolver|rocsparse|magma|"
                     r"torch_hip|aotriton|hiprtc|hipfft|rocfft|hipsolver|hipsparse|roctracer|rocprofiler)")


def check_libraries():
    """Every mapped GPU library must live inside the image. Singularity --rocm binds the host's ROCm 6.3.0 with the same sonames."""
    roots = tuple(os.path.realpath(p) + os.sep for p in (W, "/opt/rocm", "/opt/python", "/usr/lib", "/lib"))
    loaded, outside = [], []
    for line in read("/proc/self/maps").splitlines():
        parts = line.split()
        if len(parts) < 6 or not parts[-1].startswith("/"):
            continue
        path = os.path.realpath(parts[-1])
        if path in loaded or not GPU_LIB.search(os.path.basename(path)):
            continue
        loaded.append(path)
        if not path.startswith(roots):
            outside.append(path)
    lines = ["%d GPU libraries mapped, all from inside the image" % len(loaded)] if not outside else \
            ["LOADED FROM OUTSIDE THE IMAGE (run without --rocm, and without a host LD_LIBRARY_PATH):"] + outside
    lines.append("e.g. " + ", ".join(sorted(os.path.basename(p) for p in loaded)[:6]))
    return section("libraries", not outside and bool(loaded), *lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gpu", action="store_true", help="tree and sources (and weights, if given) only: the build runs this")
    ap.add_argument("--gcds", default="", help="the GCD indices to test (default: all visible)")
    a = ap.parse_args()

    say("openfold3 check -- %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    check_tree()
    check_sources()
    root = os.environ.get("OF3_WEIGHTS", "")
    if root and os.path.isdir(root):
        check_weights(root)
    if not a.no_gpu:
        check_devices(a.gcds)
        check_libraries()

    bad = [n for n, ok in RESULTS if not ok]
    say("\n%s: %s" % ("CHECK FAILED" if bad else "CHECK PASS",
                      ", ".join("%s %s" % (n, "pass" if ok else "FAIL") for n, ok in RESULTS)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
