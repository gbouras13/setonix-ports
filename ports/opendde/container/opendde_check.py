#!/usr/bin/env python
"""opendde_check.py -- is this image whole, is it running on its own libraries, and does it still reproduce the native port byte for byte?

    opendde check [--no-gpu] [--weights DIR] [--patch-tests] [--predict] [--keep DIR] [--gcds 0,1]

Sections, each PASS or FAIL, and a verdict line at the end:
  tree         the interpreter, torch, numpy, opendde, the kit's commit, the kit's packages loaded from the image, the patches, and the
               runtime settings this port depends on (HIP_LAUNCH_BLOCKING=1, LAYERNORM_TYPE=torch)
  pins         the kit's own stock/check_pins.py: opendde 1.1.1 installed at its pin, file for file
  patch-tests  (--patch-tests; CPU only) the port's tests of its eight opt-in fixes (tests/test_*.py): each wrapper against the stock
               statement it wraps, and each hook installing in fresh interpreters. The image build runs this section
  weights      (with weights) every file of the weights root against the kit's stock/PINS.json, without fetching anything
  devices      the GCDs torch sees, their architecture, and a matmul on each against a CPU reference
  libraries    every loaded GPU library comes from inside the image. Singularity's --rocm binds the HOST's ROCm 6.3.0 over the image's
               7.2 (same sonames) and that is a silent wrong answer, so it is a failure here, not a warning
  predict      (--predict, needs weights) 600 tokens on one GCD under the kit's deterministic recipe (--det 1), through the stock line
               and through the patched line; each structure must be byte-identical to tests/det600_stock.cif, the native port's
               reference (af3-setonix/opendde/golden/det600_stock, md5 bb6788dbfd2030283a3ba2adb8773155)

Exit 0 when every section selected passes, 1 otherwise. Reads nothing outside the image but the weights and the --keep directory.
"""
import argparse
import filecmp
import glob
import importlib.metadata as md
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

W = os.environ.get("ODDE_ROOT", "/opt/opendde")
KIT = os.environ.get("MODEL_OPT", os.path.join(W, "kit", "opendde"))
GOLDEN = os.path.join(W, "tests", "det600_stock.cif")
QUERY = os.path.join(W, "tests", "q600.json")
TESTS = ["test_dense_trunk_patch.py", "test_patch_hook.py", "test_chunk_floor.py", "test_tp_blocks.py", "test_native_alloc.py",
         "test_ablock.py", "test_attn_qblock.py", "test_apb_rows.py"]
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
        lines.append("opendde %s" % md.version("opendde"))
    except md.PackageNotFoundError:
        lines.append("opendde is NOT installed")
        ok = False
    for want in ("opt_core", "opendde_opt"):
        spec = importlib.util.find_spec(want)
        origin = getattr(spec, "origin", None)
        lines.append("%-12s %s" % (want, origin or "ABSENT"))
        if not origin or not os.path.realpath(origin).startswith(os.path.realpath(W) + os.sep):
            ok = False
    for f in (os.path.join(KIT, "run.sh"), os.path.join(KIT, "stock", "PINS.json"), os.path.join(KIT, "stock", "check_pins.py"),
              os.path.join(W, "scripts", "odde-pred.sh"), os.path.join(W, "scripts", "hooks", "sitecustomize.py"),
              os.path.join(W, "patches", "hook7", "sitecustomize.py"), GOLDEN, QUERY):
        if not os.path.exists(f):
            lines.append("MISSING %s" % f)
            ok = False
    blocking, ln = os.environ.get("HIP_LAUNCH_BLOCKING"), os.environ.get("LAYERNORM_TYPE")
    lines.append("HIP_LAUNCH_BLOCKING=%s  LAYERNORM_TYPE=%s" % (blocking, ln))
    if blocking != "1":
        lines.append("HIP_LAUNCH_BLOCKING must be 1 here: with 0, rank 7 segfaulted in the diffusion roll-out (2 of 2 runs, 4,096 tokens)")
        ok = False
    if ln != "torch":
        lines.append("LAYERNORM_TYPE must be torch: the fused LayerNorm needs nvcc")
        ok = False
    return section("tree", ok, *lines)


# ---------------------------------------------------------------------------------------------------------------- pins
def check_pins():
    rc, out = run([sys.executable, "-I", os.path.join(KIT, "stock", "check_pins.py"), "--quiet"], 300)
    tail = [ln for ln in out.splitlines() if ln.strip()][-2:] or ["(no output: the pin holds)"]
    return section("pins", rc == 0, "check_pins.py --quiet rc=%d" % rc, *tail)


# ---------------------------------------------------------------------------------------------------------------- patch-tests
def check_patch_tests():
    lines, ok = [], True
    for t in TESTS:
        env = dict(os.environ, ODDE_ROOT=W)
        if t == "test_patch_hook.py":
            env["ODDE_HOOK_VERSION"] = "2"                     # v1 is not in the image; hook2 is the base of the shipped chain
        t0 = time.time()
        try:
            rc, out = run([sys.executable, os.path.join(W, "tests", t)], 1800, env)
        except subprocess.TimeoutExpired:
            rc, out = 124, "timed out after 1800 s"
        last = [ln for ln in out.splitlines() if ln.strip()][-1:] or ["(no output)"]
        lines.append("%-26s rc=%d %5.0f s  %s" % (t, rc, time.time() - t0, last[0][:120]))
        ok = ok and rc == 0
    return section("patch-tests", ok, *lines)


# ---------------------------------------------------------------------------------------------------------------- weights
def check_weights():
    root = os.environ.get("ODDE_WEIGHTS", "")
    t0 = time.time()
    try:
        import opendde_opt.weights as wt
        res = wt.check(root, wt.pinned_files(KIT), KIT)
    except Exception as e:                                                  # noqa: BLE001
        return section("weights", False, "weights dir %s: the kit's check raised %r" % (root, e))
    unknown = res.get("unknown") or []
    lines = ["weights dir %s: %s (%.0f s; digests memoised in %s)"
             % (root, res.get("status"), time.time() - t0, os.environ.get("MODEL_OPT_WEIGHTS_DIGEST_DIR"))]
    lines += ["%s: sha256 %s is not the pin's (absent = missing; opendde fetch-weights DIR fetches it)" % (u.get("local"), u.get("sha256"))
              for u in unknown]
    return section("weights", res.get("status") == "pinned" and not unknown, *lines)


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
    """600 tokens, one GCD, --det 1, through the stock and the patched line: each byte-identical to the native reference."""
    out = keep or tempfile.mkdtemp(prefix="odde-check-")
    os.makedirs(out, exist_ok=True)
    lines, ok = [], True
    for line in ("stock", "patched"):
        d = os.path.join(out, line)
        shutil.rmtree(d, ignore_errors=True)
        env = dict(os.environ, ODDE_LOCAL=os.path.join(os.environ.get("ODDE_LOCAL", out), "check-" + line))
        t0 = time.time()
        cmd = ["bash", os.path.join(W, "scripts", "odde-pred.sh"), "--" + line, "-i", QUERY, "-o", d,
               "--use_msa", "false", "--use_template", "false", "--sample", "1", "--det", "1"]
        try:
            rc, text = run(cmd, 3600, env)
        except subprocess.TimeoutExpired:
            rc, text = 124, "timed out after 3600 s"
        with open(os.path.join(out, line + ".log"), "w") as fh:
            fh.write(text)
        cifs = sorted(glob.glob(os.path.join(d, "**", "*_sample_0.cif"), recursive=True))
        same = bool(cifs) and filecmp.cmp(cifs[0], GOLDEN, shallow=False)
        lines.append("%-7s rc=%d %5.0f s  %s" % (line, rc, time.time() - t0,
                     "BYTE-IDENTICAL to the native reference" if same else
                     ("DIFFERS from the native reference: " + cifs[0]) if cifs else "no structure written (see %s.log)" % line))
        ok = ok and rc == 0 and same
    if keep is None:
        shutil.rmtree(out, ignore_errors=True)
    else:
        lines.append("outputs and logs kept in " + out)
    return section("predict", ok, *lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gpu", action="store_true", help="tree, pins (and patch-tests, weights, if asked) only: the build runs this")
    ap.add_argument("--patch-tests", action="store_true", help="run the port's CPU tests of its eight opt-in fixes")
    ap.add_argument("--predict", action="store_true", help="also predict 600 tokens through stock and patched, byte-compared")
    ap.add_argument("--gcds", default="", help="the GCD indices to test (default: all visible)")
    ap.add_argument("--keep", default=None, help="--predict: keep its outputs and logs in this directory")
    a = ap.parse_args()

    say("opendde check -- %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    check_tree()
    check_pins()
    if a.patch_tests:
        check_patch_tests()
    have_weights = bool(os.environ.get("ODDE_WEIGHTS"))
    if have_weights:
        check_weights()
    if not a.no_gpu:
        check_devices(a.gcds)
        check_libraries()
        if a.predict:
            if have_weights:
                check_predict(a.keep)
            else:
                section("predict", False, "--predict needs the weights: --weights DIR or ODDE_WEIGHTS=DIR")

    bad = [n for n, ok in RESULTS if not ok]
    say("\n%s: %s" % ("CHECK FAILED" if bad else "CHECK PASS",
                      ", ".join("%s %s" % (n, "pass" if ok else "FAIL") for n, ok in RESULTS)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
