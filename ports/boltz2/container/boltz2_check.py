#!/usr/bin/env python
"""boltz2_check.py -- is this image whole, is it running on its own libraries, and does `exact` still reproduce stock byte for byte?

    boltz2 check [--no-gpu] [--weights DIR] [--predict] [--keep DIR] [--gcds 0,1]

Sections, each PASS or FAIL, and a verdict line at the end:
  tree        the interpreter, torch, numpy, boltz, the kit's commit, the kit's packages loaded from the image, the nvidia-smi shim, and
              the two runtime settings this port depends on (HIP_LAUNCH_BLOCKING=1, DEBUG_CLR_GRAPH_PACKET_CAPTURE=0)
  pins        the kit's own stock/check_pins.py: boltz 2.2.1 installed at its pin, file for file
  weights     (with weights) the four files staged as a BOLTZ_CACHE and checked by the kit's own weights verb against stock/PINS.json
  devices     the GCDs torch sees, their architecture, what the nvidia-smi shim reports, and a matmul on each against a CPU reference
  libraries   every loaded GPU library comes from inside the image. Singularity's --rocm binds the HOST's ROCm 6.3.0 over the image's
              7.2 (same sonames) and that is a silent wrong answer, so it is a failure here, not a warning
  predict     (--predict, needs weights) the kit's 1BRS x2 example (398 tokens, seed 1) through stock and through the kit's exact mode,
              compared byte for byte. exact replays a captured sampler graph, so a pass also shows DEBUG_CLR_GRAPH_PACKET_CAPTURE=0 in force

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
import tempfile
import time

W = os.environ.get("B2S_ROOT", "/opt/boltz2")
KIT = os.environ.get("MODEL_OPT", os.path.join(W, "kit", "boltz2"))
EXAMPLE = os.path.join(KIT, "inputs", "1BRS_x2_barnase_barstar.yaml")
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
        lines.append("boltz  %s" % md.version("boltz"))
    except md.PackageNotFoundError:
        lines.append("boltz is NOT installed")
        ok = False
    for want in ("opt_core", "boltz2_opt"):
        spec = importlib.util.find_spec(want)
        origin = getattr(spec, "origin", None)
        lines.append("%-12s %s" % (want, origin or "ABSENT"))
        if not origin or not os.path.realpath(origin).startswith(os.path.realpath(W) + os.sep):
            ok = False
    smi = shutil.which("nvidia-smi")
    lines.append("nvidia-smi   %s" % (smi or "ABSENT"))
    if not smi or not os.path.realpath(smi).startswith(os.path.realpath(W) + os.sep):
        lines.append("the kit identifies the GPU with nvidia-smi alone: the port's shim must come first on PATH")
        ok = False
    for f in (EXAMPLE, os.path.join(KIT, "run.sh"), os.path.join(KIT, "stock", "PINS.json"),
              os.path.join(W, "scripts", "b2s-pred.sh"), os.path.join(W, "scripts", "b2s_launch.py")):
        if not os.path.exists(f):
            lines.append("MISSING %s" % f)
            ok = False
    blocking, capture = os.environ.get("HIP_LAUNCH_BLOCKING"), os.environ.get("DEBUG_CLR_GRAPH_PACKET_CAPTURE")
    lines.append("HIP_LAUNCH_BLOCKING=%s  DEBUG_CLR_GRAPH_PACKET_CAPTURE=%s" % (blocking, capture))
    if blocking != "1":
        lines.append("HIP_LAUNCH_BLOCKING must be 1 here: stock Boltz-2 itself segfaults above ~500 tokens without it")
        ok = False
    if capture != "0":
        lines.append("DEBUG_CLR_GRAPH_PACKET_CAPTURE must be 0 on ROCm 7.2: a replayed graph corrupts the host heap (fast's roll-out)")
        ok = False
    return section("tree", ok, *lines)


# ---------------------------------------------------------------------------------------------------------------- pins
def check_pins():
    env = dict(os.environ)
    env.pop("BOLTZ2_OPT", None)
    rc, out = run([sys.executable, "-I", os.path.join(KIT, "stock", "check_pins.py"), "--quiet"], 300, env)
    tail = [ln for ln in out.splitlines() if ln.strip()][-2:] or ["(no output: the pins hold)"]
    return section("pins", rc == 0, "check_pins.py --quiet rc=%d" % rc, *tail)


# ---------------------------------------------------------------------------------------------------------------- weights
def stage_cache():
    rc, out = run(["bash", os.path.join(W, "scripts", "stage-cache.sh")], 1800)
    lines = [ln for ln in out.splitlines() if ln.strip()]
    return (lines[-1] if rc == 0 and lines else None), lines


def check_weights():
    cache, lines = stage_cache()
    if not cache:
        return section("weights", False, *(lines[-3:] or ["stage-cache.sh failed"]))
    env = dict(os.environ, BOLTZ_CACHE=cache)
    env.pop("BOLTZ2_OPT", None)
    t0 = time.time()
    rc, out = run([sys.executable, "-m", "boltz2_opt.weights", cache], 3600, env)
    tail = [ln for ln in out.splitlines() if ln.strip()][-3:]
    return section("weights", rc == 0, "weights dir %s, staged at %s" % (os.environ.get("B2S_WEIGHTS"), cache),
                   "the kit's weights verb rc=%d (%.0f s; digests memoised in %s)"
                   % (rc, time.time() - t0, os.environ.get("MODEL_OPT_WEIGHTS_DIGEST_DIR")), *tail)


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
        lines.append("the shim must report every visible GCD: the kit refuses every mode otherwise")
        ok = False
    return section("devices", ok, *lines)


# ---------------------------------------------------------------------------------------------------------------- libraries
GPU_LIB = re.compile(r"lib(amdhip|hsa-runtime|rocblas|hipblaslt|MIOpen|rocrand|hiprand|rccl|amd_comgr|rocsolver|rocsparse|magma|"
                     r"torch_hip|aotriton|hiprtc|hipfft|rocfft|hipsolver|hipsparse|roctracer|rocprofiler)")


def check_libraries():
    """Every mapped GPU library must live inside the image. Singularity --rocm binds the host's ROCm 6.3.0 with the same sonames."""
    inside = tuple(os.path.realpath(p) + os.sep for p in (W, "/opt/rocm", "/opt/python", "/usr/lib", "/lib"))
    loaded, outside = [], []
    for line in read("/proc/self/maps").splitlines():
        parts = line.split()
        if len(parts) < 6 or not parts[-1].startswith("/"):
            continue
        path = os.path.realpath(parts[-1])
        if path in loaded or not GPU_LIB.search(os.path.basename(path)):
            continue
        loaded.append(path)
        if not path.startswith(inside):
            outside.append(path)
    lines = ["%d GPU libraries mapped, all from inside the image" % len(loaded)] if not outside else \
            ["LOADED FROM OUTSIDE THE IMAGE (run without --rocm):"] + outside
    lines.append("e.g. " + ", ".join(sorted(os.path.basename(p) for p in loaded)[:6]))
    return section("libraries", not outside and bool(loaded), *lines)


# ---------------------------------------------------------------------------------------------------------------- predict
def check_predict(keep=None):
    """1BRS x2 (398 tokens, seed 1, upstream defaults) through stock and the kit's exact mode; model bytes compared."""
    out = keep or tempfile.mkdtemp(prefix="b2s-check-")
    os.makedirs(out, exist_ok=True)
    lines, ok, dirs = [], True, {}
    for mode in ("off", "exact"):
        d = os.path.join(out, mode)
        env = dict(os.environ, B2S_LOCAL=os.path.join(os.environ.get("B2S_LOCAL", out), "check-" + mode))
        env.pop("BOLTZ_CACHE", None)
        t0 = time.time()
        rc, text = run(["bash", os.path.join(W, "scripts", "b2s-pred.sh"), "--mode", mode, "--input", EXAMPLE,
                        "--out_dir", d, "--seed", "1"], 3600, env)
        with open(os.path.join(out, mode + ".log"), "w") as fh:
            fh.write(text)
        dirs[mode] = d
        tail = [ln for ln in text.splitlines() if "EXIT mode=" in ln][-1:] or \
               [ln for ln in text.splitlines() if ln.strip()][-1:] or ["(no output)"]
        lines.append("%-5s rc=%d  %5.0f s  %s" % (mode, rc, time.time() - t0, tail[0][:150]))
        ok = ok and rc == 0
    if ok:
        rc, text = run([sys.executable, os.path.join(W, "tools", "b2s_outputs.py"), "compare", dirs["off"], dirs["exact"]], 600)
        rows = [ln.strip() for ln in text.splitlines() if "bytes_identical=" in ln]
        same = bool(rows) and all("bytes_identical=True" in r for r in rows)
        lines += [r[:170] for r in rows[:3]]
        lines.append("exact is byte-identical to stock" if same else "exact and stock DIFFER (or no model was written)")
        ok = rc == 0 and same
    if keep is None:
        shutil.rmtree(out, ignore_errors=True)
    else:
        lines.append("outputs and logs kept in " + out)
    return section("predict", ok, *lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gpu", action="store_true", help="tree and pins (and weights, if given) only: the build runs this")
    ap.add_argument("--predict", action="store_true", help="also predict 1BRS x2 through stock and exact and compare the bytes")
    ap.add_argument("--gcds", default="", help="the GCD indices to test (default: all visible)")
    ap.add_argument("--keep", default=None, help="--predict: keep its outputs and logs in this directory")
    a = ap.parse_args()

    say("boltz2 check -- %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    check_tree()
    check_pins()
    have_weights = bool(os.environ.get("B2S_WEIGHTS"))
    if have_weights:
        check_weights()
    if not a.no_gpu:
        check_devices(a.gcds)
        check_libraries()
        if a.predict:
            if have_weights:
                check_predict(a.keep)
            else:
                section("predict", False, "--predict needs the weights: --weights DIR or B2S_WEIGHTS=DIR")

    bad = [n for n, ok in RESULTS if not ok]
    say("\n%s: %s" % ("CHECK FAILED" if bad else "CHECK PASS",
                      ", ".join("%s %s" % (n, "pass" if ok else "FAIL") for n, ok in RESULTS)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
