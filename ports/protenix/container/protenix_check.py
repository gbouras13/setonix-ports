#!/usr/bin/env python
"""protenix_check.py -- is this image whole, is it running on its own libraries, and is the one-GCD line still stock's arithmetic?

    protenix check [--no-gpu] [--weights DIR] [--predict] [--keep DIR]

Sections, each PASS or FAIL, and a verdict line at the end:
  tree        the interpreter, torch, numpy, protenix, the kit's commit, the kit's packages and the port's start-up hooks loaded from the
              image, the C compiler and Python headers Triton builds its kernel launchers with, and the settings this port depends on
              (PROTENIX_OPT_FORCE=1, LAYERNORM_TYPE=torch, the three levers that cannot run here dropped, HIP_LAUNCH_BLOCKING=1)
  pins        the kit's own stock/check_pins.py: protenix 2.0.0 installed at its pin, file for file
  weights     (with weights) the seven files checked by the kit's own routine against stock/PINS.json; nothing is downloaded
  devices     the GCDs torch sees, their architecture, and a matmul on each against a CPU reference
  libraries   every loaded GPU library comes from inside the image. Singularity's --rocm binds the HOST's ROCm 6.3.0 over the image's
              7.2 (same sonames) and that is a silent wrong answer, so it is a failure here, not a warning
  predict     (--predict, needs weights) the kit's 1BRS example (barnase:barstar, 199 tokens, its precomputed MSAs; stock defaults: 10
              cycles, 200 steps, 5 samples, seed 101) through stock's own CLI and through the one-GCD big line, both under deterministic
              algorithms (PXS_DETERMINISTIC=1, CUBLAS_WORKSPACE_CONFIG=:4096:8), compared byte for byte (jobs 50030425, 50379609: identical)

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

W = os.environ.get("PXS_ROOT", "/opt/protenix")
KIT = os.environ.get("MODEL_OPT", os.path.join(W, "kit", "protenix_v2"))
FPF = os.path.join(KIT, "opt", "forward", "flashpairformer")
EXAMPLE = os.path.join(FPF, "public_inputs", "p199.json")
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
    try:
        v = md.version("protenix")
        lines.append("protenix %s" % v)
        ok = ok and v == "2.0.0"
    except md.PackageNotFoundError:
        lines.append("protenix is NOT installed")
        ok = False
    for want in ("opt_core", "protenix_opt"):
        spec = importlib.util.find_spec(want)
        origin = getattr(spec, "origin", None)
        lines.append("%-14s %s" % (want, origin or "ABSENT"))
        ok = ok and inside(origin)
    purelib = sysconfig.get_paths()["purelib"]
    for hook in ("setonix_rankcache.pth", "setonix_rankcache.py", "setonix_chunkcap.py", "setonix_det.py"):
        if not os.path.isfile(os.path.join(purelib, hook)):
            lines.append("MISSING start-up hook %s in %s" % (hook, purelib))
            ok = False
    lines.append("start-up hooks in %s" % purelib)
    cc = shutil.which(os.environ.get("CC") or "gcc") or shutil.which("cc")
    header = os.path.join(sysconfig.get_paths()["include"], "Python.h")
    lines.append("C compiler %s, Python.h %s (Triton builds its kernel launchers with them)"
                 % (cc or "ABSENT", "present" if os.path.isfile(header) else "ABSENT"))
    ok = ok and bool(cc) and os.path.isfile(header)
    for f in (EXAMPLE, os.path.join(KIT, "run.sh"), os.path.join(KIT, "stock", "PINS.json"),
              os.path.join(KIT, "stock", "check_pins.py"), os.path.join(W, "scripts", "ptx-env.sh"),
              os.path.join(W, "scripts", "ptx-pred.sh"), os.path.join(W, "tools", "ptx_outputs.py")):
        if not os.path.exists(f):
            lines.append("MISSING %s" % f)
            ok = False
    env = os.environ
    levers = set((env.get("MODEL_OPT_LEVERS_OFF") or "").split(","))
    lines.append("PROTENIX_OPT_FORCE=%s LAYERNORM_TYPE=%s HIP_LAUNCH_BLOCKING=%s MODEL_OPT_LEVERS_OFF=%s"
                 % (env.get("PROTENIX_OPT_FORCE"), env.get("LAYERNORM_TYPE"), env.get("HIP_LAUNCH_BLOCKING"),
                    env.get("MODEL_OPT_LEVERS_OFF")))
    if env.get("PROTENIX_OPT_FORCE") != "1":
        lines.append("PROTENIX_OPT_FORCE must be 1: the kit identifies the GPU with nvidia-smi alone and refuses every mode without it")
        ok = False
    if env.get("LAYERNORM_TYPE") != "torch":
        lines.append("LAYERNORM_TYPE must be torch: stock JIT-builds a CUDA LayerNorm with nvcc otherwise")
        ok = False
    if env.get("HIP_LAUNCH_BLOCKING") != "1":
        lines.append("HIP_LAUNCH_BLOCKING should be 1 here (rocBLAS's gfx90a gap; every validated run used it)")
        ok = False
    if not {"fastln_prebuilt", "ln_core", "sampler_fuse"} <= levers:
        lines.append("MODEL_OPT_LEVERS_OFF must name fastln_prebuilt, ln_core and sampler_fuse: none can run here, and a mode refuses a "
                     "partial lever set")
        ok = False
    return section("tree", ok, *lines)


# ---------------------------------------------------------------------------------------------------------------- pins
def check_pins():
    env = dict(os.environ)
    env.pop("PROTENIX_OPT", None)
    rc, out = run([sys.executable, "-I", os.path.join(KIT, "stock", "check_pins.py"), "--quiet"], 300, env)
    tail = [ln for ln in out.splitlines() if ln.strip()][-2:] or ["(no output: the pins hold)"]
    return section("pins", rc == 0, "check_pins.py --quiet rc=%d" % rc, *tail)


# ---------------------------------------------------------------------------------------------------------------- weights
def check_weights(root):
    code = ("import sys; from protenix_opt.weights import check; from protenix_opt.stack import pins; "
            "sys.exit(1 if check(sys.argv[1], pins()) else 0)")
    env = dict(os.environ)
    env.pop("PROTENIX_OPT", None)
    t0 = time.time()
    rc, out = run([sys.executable, "-c", code, root], 1800, env)
    tail = [ln for ln in out.splitlines() if ln.strip()][-3:]
    return section("weights", rc == 0, "weights dir %s" % root,
                   "the kit's own check rc=%d (%.0f s; its digest memo in %s)" % (rc, time.time() - t0, os.environ.get("PROTENIX_OPT_CACHE_DIR")),
                   *tail)


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


# ---------------------------------------------------------------------------------------------------------------- predict
ARGS = ["--seeds", "101"]                                      # stock's defaults for everything else: 10 cycles, 200 steps, 5 samples


def gcd_plan(n_arms):
    """One GCD per arm when the parent's view allows pinning them; None runs the arms one after another in the parent's view."""
    env = os.environ
    if env.get("HIP_VISIBLE_DEVICES") or env.get("CUDA_VISIBLE_DEVICES"):
        return None
    rocr = [g for g in (env.get("ROCR_VISIBLE_DEVICES") or "").split(",") if g.strip()]
    if not rocr:
        import torch
        rocr = [str(i) for i in range(torch.cuda.device_count())]
    return rocr[:n_arms] if len(rocr) >= n_arms else None


def check_predict(keep=None):
    """1BRS at stock defaults through stock's CLI and the one-GCD big line, deterministic, outputs compared byte for byte."""
    out = keep or tempfile.mkdtemp(prefix="pxs-check-")
    os.makedirs(out, exist_ok=True)
    query = read(EXAMPLE).replace("__KIT__", FPF)
    arms = (("stock", ["--mode", "stock"]), ("big", []))       # ptx-pred.sh routes: stock's own CLI | the kit's big line, one GCD
    plan = gcd_plan(len(arms))
    lines, ok, procs = [], True, []
    for k, (label, route) in enumerate(arms):
        d = os.path.join(out, label)
        os.makedirs(os.path.join(d, "in"), exist_ok=True)
        q = os.path.join(d, "in", "p199.json")
        with open(q, "w") as fh:
            fh.write(query)
        env = dict(os.environ, PXS_DETERMINISTIC="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                   PXS_LOCAL=os.path.join(os.environ.get("PXS_LOCAL", out), "check-" + label))
        # without MODEL_OPT, ptx-pred.sh sources ptx-env.sh again, and each arm keys its own caches under its own PXS_LOCAL
        for v in ("MODEL_OPT", "PXS_DUMP_PRED", "MODEL_OPT_JIT_ROOT", "TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR"):
            env.pop(v, None)
        if plan:
            env["ROCR_VISIBLE_DEVICES"] = plan[k]
        cmd = ["bash", os.path.join(W, "scripts", "ptx-pred.sh")] + route + ["-i", q, "-o", os.path.join(d, "out")] + ARGS
        log = open(os.path.join(out, label + ".log"), "w")
        procs.append((label, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), time.time(), log))
        if not plan:                                           # one GCD visible: one arm at a time
            procs[-1][1].wait()
    for label, p, t0, log in procs:
        try:
            rc = p.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            p.kill()
            rc = "timeout"
        log.close()
        text = read(os.path.join(out, label + ".log"))
        det = "deterministic=True" in text and "use_deterministic_algorithms(True" in text
        tail = [ln for ln in text.splitlines() if "[ptx-pred]" in ln][-1:] or [ln for ln in text.splitlines() if ln.strip()][-1:] or ["(no output)"]
        lines.append("%-5s rc=%s %5.0f s gcd=%s deterministic=%s  %s"
                     % (label, rc, time.time() - t0, plan[[a for a, _ in arms].index(label)] if plan else "shared", det, tail[0][:140]))
        ok = ok and rc == 0 and det
    if ok:
        rc, text = run([sys.executable, os.path.join(W, "tools", "ptx_outputs.py"), "compare", "--identical",
                        os.path.join(out, "stock", "out"), os.path.join(out, "big", "out")], 600)
        rows = [ln.strip() for ln in text.splitlines() if "bytes_identical=" in ln or "byte-identical" in ln]
        lines += [r[:170] for r in rows[-6:]]
        lines.append("the one-GCD big line is byte-identical to stock" if rc == 0 else "big and stock DIFFER (or a sample is missing)")
        ok = rc == 0
    if keep is None:
        shutil.rmtree(out, ignore_errors=True)
    else:
        lines.append("outputs and logs kept in " + out)
    return section("predict", ok, *lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gpu", action="store_true", help="tree and pins (and weights, if given) only: the build runs this")
    ap.add_argument("--predict", action="store_true", help="also predict 1BRS through stock and the one-GCD big line and compare the bytes")
    ap.add_argument("--gcds", default="", help="the GCD indices to test (default: all visible)")
    ap.add_argument("--keep", default=None, help="--predict: keep its outputs and logs in this directory")
    a = ap.parse_args()

    say("protenix check -- %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    check_tree()
    check_pins()
    root = os.environ.get("PROTENIX_ROOT_DIR", "")
    have_weights = bool(root) and os.path.isdir(root)
    if have_weights:
        check_weights(root)
    if not a.no_gpu:
        check_devices(a.gcds)
        check_libraries()
        if a.predict:
            if have_weights:
                check_predict(a.keep)
            else:
                section("predict", False, "--predict needs the weights: --weights DIR or PROTENIX_ROOT_DIR=DIR")

    bad = [n for n, ok in RESULTS if not ok]
    say("\n%s: %s" % ("CHECK FAILED" if bad else "CHECK PASS",
                      ", ".join("%s %s" % (n, "pass" if ok else "FAIL") for n, ok in RESULTS)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
