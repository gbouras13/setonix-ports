#!/usr/bin/env python
"""esmfold2_check.py -- is this image whole, is it running on its own libraries, and does `exact` still reproduce stock byte for byte?

    esmfold2 check [--no-gpu] [--weights DIR] [--predict] [--keep DIR] [--gcds 0,1]

Sections, each PASS or FAIL, and a verdict line at the end:
  tree        the interpreter, torch, numpy, esm, transformers, the kit's commit, the kit's packages (and its autoload hook) loaded
              from the image, the nvidia-smi shim, the boot-fix directory, and DEBUG_CLR_GRAPH_PACKET_CAPTURE=0
  pins        the kit's own stock/check_pins.py through scripts/efs_pins.py: esm and the transformers fork at their pinned commits (the
              CUDA accelerated layer is reported absent, as it is on every ROCm stack)
  weights     (with weights) the kit's own weights verb: every file of the three snapshots against stock/PINS.json (sha256)
  devices     the GCDs torch sees, their architecture, what the nvidia-smi shim reports, and a matmul on each against a CPU reference
  libraries   every loaded GPU library comes from inside the image. Singularity's --rocm binds the HOST's ROCm 6.3.0 over the image's
              7.2 (same sonames) and that is a silent wrong answer, so it is a failure here, not a warning
  predict     (--predict, needs weights) the 1BRS barstar + barnase example (199 tokens, seed 0, --det 1) through stock-fused and
              through the kit's exact mode (scripts/ef2.sh), compared byte for byte. exact replays captured graphs under the port's
              settings, so a pass also shows them in force

Exit 0 when every section selected passes, 1 otherwise. Reads nothing outside the image but the weights and the --keep directory.
"""
import argparse
import importlib.metadata as md
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time

W = os.environ.get("EFS_ROOT", "/opt/esmfold2")
KIT = os.environ.get("MODEL_OPT", os.path.join(W, "kit", "esmfold2"))
EXAMPLE = os.path.join(W, "inputs", "smoke_1brs.json")
RESULTS = []


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


def run(cmd, timeout, env=None):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    return p.returncode, (p.stdout + p.stderr)


def inside(path):
    return bool(path) and os.path.realpath(path).startswith(os.path.realpath(W) + os.sep)


# ---------------------------------------------------------------------------------------------------------------- tree
def check_tree():
    import numpy
    lines = ["python %s (%s)" % (sys.version.split()[0], sys.executable),
             "numpy  %s" % numpy.__version__,
             read(os.path.join(W, "source.txt"), "source unknown"),
             read(os.path.join(W, "kit.txt"), "kit unknown")]
    ok = True
    try:
        import torch
        lines.append("torch  %s (hip %s, cuda %s)" % (torch.__version__, torch.version.hip, torch.version.cuda))
        if not torch.version.hip:
            lines.append("NOT a ROCm build of torch")
            ok = False
    except Exception as e:                                                  # noqa: BLE001
        lines.append("torch does not import: %r" % (e,))
        ok = False
    for dist, want in (("esm", "3.3.0"), ("transformers", "4.57.6")):
        try:
            got = md.version(dist)
            lines.append("%-12s %s" % (dist, got))
            ok = ok and got == want
        except md.PackageNotFoundError:
            lines.append("%s is NOT installed" % dist)
            ok = False
    for want in ("opt_core", "esmfold2_opt"):
        spec = importlib.util.find_spec(want)
        origin = getattr(spec, "origin", None)
        lines.append("%-12s %s" % (want, origin or "ABSENT"))
        ok = ok and inside(origin)
    pth = os.path.join(sysconfig.get_paths()["purelib"], "esmfold2_opt_autoload.pth")
    lines.append("autoload     %s" % (pth if os.path.exists(pth) else "ABSENT (" + pth + ")"))
    ok = ok and os.path.exists(pth)
    smi = shutil.which("nvidia-smi")
    lines.append("nvidia-smi   %s" % (smi or "ABSENT"))
    if not inside(smi):
        lines.append("the kit identifies the GPU with nvidia-smi: the port's shim must come first on PATH")
        ok = False
    boot = os.path.join(W, "scripts", "boot")
    if boot not in os.environ.get("PYTHONPATH", "").split(":"):
        lines.append("the boot-fix directory %s is not on PYTHONPATH (scripts/ef2-env.sh puts it there)" % boot)
        ok = False
    for f in (EXAMPLE, os.path.join(KIT, "run.sh"), os.path.join(KIT, "stock", "PINS.json"), os.path.join(boot, "sitecustomize.py"),
              os.path.join(W, "scripts", "ef2.sh"), os.path.join(W, "scripts", "ef2-run.sh"), os.path.join(W, "scripts", "ef2-stage.sh")):
        if not os.path.exists(f):
            lines.append("MISSING %s" % f)
            ok = False
    capture = os.environ.get("DEBUG_CLR_GRAPH_PACKET_CAPTURE")
    lines.append("DEBUG_CLR_GRAPH_PACKET_CAPTURE=%s (HIP_LAUNCH_BLOCKING=1 and the allocator setting are per route: scripts/ef2.sh)"
                 % capture)
    if capture != "0":
        lines.append("DEBUG_CLR_GRAPH_PACKET_CAPTURE must be 0 here (scripts/ef2-env.sh)")
        ok = False
    return section("tree", ok, *lines)


# ---------------------------------------------------------------------------------------------------------------- pins
def check_pins():
    env = dict(os.environ)
    env.pop("ESMFOLD2_OPT", None)
    rc, out = run([sys.executable, "-I", os.path.join(W, "scripts", "efs_pins.py"), KIT], 300, env)
    tail = [ln.strip()[:200] for ln in out.splitlines() if ln.strip()][-2:] or ["(no output: the forks are at their pins)"]
    return section("pins", rc == 0, "efs_pins.py (the kit's check_pins.py) rc=%d" % rc, *tail)


# ---------------------------------------------------------------------------------------------------------------- weights
def check_weights():
    d = os.environ.get("EFS_WEIGHTS")
    env = dict(os.environ)
    env.pop("ESMFOLD2_OPT", None)
    t0 = time.time()
    rc, out = run([sys.executable, "-m", "esmfold2_opt.weights", d], 3600, env)
    tail = [ln.strip()[:200] for ln in out.splitlines() if ln.strip()][-3:]
    return section("weights", rc == 0, "weights dir %s" % d,
                   "the kit's weights verb rc=%d (%.0f s; digests memoised in %s)"
                   % (rc, time.time() - t0, os.environ.get("ESMFOLD2_OPT_WEIGHTS_MEMO_DIR")), *tail)


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
    rc, out = run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"], 60)
    shim = [ln for ln in out.splitlines() if ln.strip()]
    lines.append("nvidia-smi shim rc=%d: %d GPU line(s), e.g. %s" % (rc, len(shim), shim[0] if shim else "none"))
    if rc != 0 or len(shim) != n:
        lines.append("the shim must report every visible GCD: the kit's dry run refuses otherwise")
        ok = False
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
            ["LOADED FROM OUTSIDE THE IMAGE (run without --rocm):"] + outside
    lines.append("e.g. " + ", ".join(sorted(os.path.basename(p) for p in loaded)[:6]))
    return section("libraries", not outside and bool(loaded), *lines)


# ---------------------------------------------------------------------------------------------------------------- predict
def check_predict(keep=None):
    """1BRS barstar + barnase (199 tokens, seed 0, --det 1, library defaults) through stock-fused and exact; the outputs compared."""
    out = keep or tempfile.mkdtemp(prefix="efs-check-", dir=os.environ.get("EFS_LOCAL"))
    os.makedirs(out, exist_ok=True)
    lines, ok, dirs = [], True, {}
    for route in ("stock-fused", "exact"):
        d = os.path.join(out, route)
        t0 = time.time()
        rc, text = run(["bash", os.path.join(W, "scripts", "ef2.sh"), route, "--variant", "fast", "--input", EXAMPLE,
                        "--out_dir", d, "--seeds", "0", "--det", "1"], 3600)
        with open(os.path.join(out, route + ".log"), "w") as fh:
            fh.write(text)
        dirs[route] = d
        folds = [ln for ln in text.splitlines() if re.search(r"fold \S+ s0 samples=1 [0-9.]+s", ln)]
        tail = folds[-1:] or [ln for ln in text.splitlines() if ln.strip()][-1:] or ["(no output)"]
        lines.append("%-11s rc=%d  %5.0f s  %s" % (route, rc, time.time() - t0, tail[0].strip()[:140]))
        ok = ok and rc == 0
    if ok:
        rc, text = run([sys.executable, os.path.join(W, "tools", "ef2_compare.py"), dirs["stock-fused"], dirs["exact"], "--quiet"], 600)
        lines += [ln.strip()[:170] for ln in text.splitlines() if ln.strip()][-1:]
        lines.append("exact is byte-identical to stock-fused" if rc == 0 else "exact and stock-fused DIFFER (or nothing was written)")
        ok = rc == 0
    if keep is None:
        shutil.rmtree(out, ignore_errors=True)
    else:
        lines.append("outputs and logs kept in " + out)
    return section("predict", ok, *lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gpu", action="store_true", help="tree and pins (and weights, if given) only: the build runs this")
    ap.add_argument("--predict", action="store_true", help="also fold 1BRS through stock-fused and exact and compare the bytes")
    ap.add_argument("--gcds", default="", help="the GCD indices to test (default: all visible)")
    ap.add_argument("--keep", default=None, help="--predict: keep its outputs and logs in this directory")
    a = ap.parse_args()

    say("esmfold2 check -- %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    check_tree()
    check_pins()
    have_weights = bool(os.environ.get("EFS_WEIGHTS"))
    if have_weights:
        check_weights()
    if not a.no_gpu:
        check_devices(a.gcds)
        check_libraries()
        if a.predict:
            if have_weights:
                check_predict(a.keep)
            else:
                section("predict", False, "--predict needs the weights: --weights DIR or EFS_WEIGHTS=DIR")

    bad = [n for n, ok in RESULTS if not ok]
    say("\n%s: %s" % ("CHECK FAILED" if bad else "CHECK PASS",
                      ", ".join("%s %s" % (n, "pass" if ok else "FAIL") for n, ok in RESULTS)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
