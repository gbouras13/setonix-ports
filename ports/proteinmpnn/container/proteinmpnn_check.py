#!/usr/bin/env python
"""proteinmpnn_check.py -- is this image whole, is it running on its own libraries, and does it still reproduce stock byte for byte?

    proteinmpnn check [--no-gpu] [--design] [--gcds 0,1,2]

Sections, each PASS or FAIL, and a verdict line at the end:
  tree        the interpreter, torch, numpy, the kit's commit, upstream's commit, and the three gfx90a fixes this port depends on
  pins        the kit's own stock/check_pins.py for both weight sets: the checkout at the pinned commit, the weights at the pinned sha256
  libraries   every loaded GPU library comes from inside the image. Singularity's --rocm binds the HOST's ROCm 6.3.0 over the image's
              7.2 (same sonames) and that is a silent wrong answer, so it is a failure here, not a warning
  devices     the GCDs torch sees, their architecture, and a matmul on each against a CPU reference
  draw        torch's uniform transform on this build is rocRAND's, not cuRAND's -- the one numerical difference the kit's own 512-draw
              probe cannot see, and the reason scripts/pms_worker_boot.py replaces the fused draw kernel (`rocm_draw`)
  design      (--design) upstream's 429-residue 3HTN through BOTH routes, stock and the kit's exact worker, compared byte for byte.
              429 residues is past the length where ROCm 7.2's graph packet capture corrupts the heap, so this also proves fix 3 is in
              force; `--no-gpu` and the build-time check skip it

Exit 0 when every section selected passes, 1 otherwise. Reads nothing outside the image but the output directory --design writes.
"""
import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

W = os.environ.get("PMS_ROOT", "/opt/proteinmpnn")
KIT = os.path.join(W, "kit", "proteinmpnn")
MPNN_DIR = os.environ.get("MPNN_DIR", os.path.join(W, "ProteinMPNN"))
EXAMPLE = os.path.join(MPNN_DIR, "inputs", "PDB_complexes", "pdbs", "3HTN.pdb")
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


# ---------------------------------------------------------------------------------------------------------------- tree
def check_tree():
    import numpy
    lines = ["python %s (%s)" % (sys.version.split()[0], sys.executable),
             "numpy  %s" % numpy.__version__,
             read(os.path.join(W, "source.txt"), "source unknown"),
             read(os.path.join(W, "kit.txt"), "kit unknown"),
             read(os.path.join(W, "mpnn.txt"), "ProteinMPNN unknown")]
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
    for want in ("opt_core", "proteinmpnn_opt"):
        spec = importlib.util.find_spec(want)
        origin = getattr(spec, "origin", None)
        lines.append("%-16s %s" % (want, origin or "ABSENT"))
        if not origin or not os.path.realpath(origin).startswith(os.path.realpath(W) + os.sep):
            ok = False
    for f in (EXAMPLE, os.path.join(MPNN_DIR, "protein_mpnn_run.py"),
              os.path.join(MPNN_DIR, "vanilla_model_weights", "v_48_020.pt"), os.path.join(KIT, "run.sh")):
        if not os.path.exists(f):
            lines.append("MISSING %s" % f)
            ok = False
    lines.append("the three gfx90a fixes: draw kernel = %s | DEBUG_CLR_GRAPH_PACKET_CAPTURE = %s | hybrid_gemm = 0 (the kit's probe refuses it here)"
                 % (os.environ.get("PMS_WORKER_PATCHES", "rocm_draw (default on a ROCm torch)"),
                    os.environ.get("DEBUG_CLR_GRAPH_PACKET_CAPTURE", "UNSET -- pms-env.sh was not sourced")))
    if os.environ.get("DEBUG_CLR_GRAPH_PACKET_CAPTURE") != "0":
        lines.append("DEBUG_CLR_GRAPH_PACKET_CAPTURE must be 0 on ROCm 7.2: a replayed graph corrupts the host heap")
        ok = False
    return section("tree", ok, *lines)


# ---------------------------------------------------------------------------------------------------------------- pins
def check_pins():
    ok, lines = True, []
    for variant in ("vanilla", "soluble"):
        p = subprocess.run([sys.executable, "-I", os.path.join(KIT, "stock", "check_pins.py"),
                            "--variant", variant, "--mpnn-dir", MPNN_DIR],
                           capture_output=True, text=True, timeout=300)
        out = (p.stdout + p.stderr).strip().splitlines()
        lines.append("%-8s rc=%d  %s" % (variant, p.returncode, out[-1] if out else "(no output)"))
        ok = ok and p.returncode == 0
    return section("pins", ok, *lines)


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
        good = d < 1e-2
        ok = ok and good
        lines.append("GCD %d  %-22s %-22s %5.1f GiB  matmul max|d| %.2e %s"
                     % (i, p.name, p.gcnArchName, p.total_memory / 2**30, d, "" if good else "<- WRONG"))
    lines.append("capability %s -- note gfx90a reports H100's (9, 0), so any capability-keyed kernel table needs checking"
                 % (torch.cuda.get_device_capability(want[0]),))
    return section("devices", ok, *lines)


# ---------------------------------------------------------------------------------------------------------------- draw
COUNTS = re.compile(r"(\d+) values; exact match curand=(\d+) rocrand=(\d+); within 1 ulp curand=(\d+) rocrand=(\d+)")


def check_draw():
    """torch on ROCm draws through rocRAND. Its uniform transform differs from cuRAND's, which is what the kit's kernel computes.

    tools/pms_drawcheck.py does the comparison (a numpy Philox4x32-10 against the device's exponential_, both transforms); this reads a
    verdict out of its report: rocRAND must account for every value within 1 ulp, cuRAND must not account for all of them.
    """
    spec = importlib.util.spec_from_file_location("pms_drawcheck", os.path.join(W, "tools", "pms_drawcheck.py"))
    dc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dc)
    lines, ok = dc.check_exponential("cuda", (0, 37, 1234), (0, 4, 1024)), True
    for ln in lines:
        m = COUNTS.search(ln)
        if not m:
            ok = False
            continue
        total, _, _, ulp_cu, ulp_roc = (int(x) for x in m.groups())
        if ulp_roc != total or ulp_cu == total:
            ok = False
    lines = list(lines) + [
        "rocRAND must match every value, cuRAND must not match all of them (the two agree to ~1e-6, which is why the kit's own",
        "512-draw probe passes here anyway); scripts/pms_worker_boot.py `rocm_draw` replaces the kit's kernel accordingly"]
    return section("draw", ok, *lines)


# ---------------------------------------------------------------------------------------------------------------- design
def check_design(keep=None):
    """3HTN (429 residues, past the graph-replay crash threshold) through stock and through the kit's exact worker; bytes compared."""
    out = keep or tempfile.mkdtemp(prefix="pms-check-")
    lines, ok, dirs = [], True, {}
    common = ["--seed", "37", "--num_seq_per_target", "4", "--sampling_temp", "0.1", "--batch_size", "1"]
    for mode in ("off", "exact"):
        d = os.path.join(out, mode)
        t0 = time.time()
        p = subprocess.run(["bash", os.path.join(W, "scripts", "pms-run.sh"), "design", "--mode", mode,
                            "--input", EXAMPLE, "--out", d] + common, capture_output=True, text=True, timeout=1800)
        dirs[mode] = d
        tail = [ln for ln in (p.stdout + p.stderr).splitlines() if ln.strip()][-1:] or ["(no output)"]
        lines.append("%-5s rc=%d  %5.1f s  %s" % (mode, p.returncode, time.time() - t0, tail[0][:110]))
        if p.returncode != 0:
            ok = False
    if ok:
        p = subprocess.run([sys.executable, os.path.join(W, "tools", "pms_compare.py"), dirs["off"], dirs["exact"]],
                           capture_output=True, text=True, timeout=600)
        lines += [ln for ln in p.stdout.splitlines() if ln.strip()][-3:]
        ok = p.returncode == 0
        lines.append("exact is byte-identical to stock" if ok else "exact and stock DIFFER -- see the lines above")
    if keep is None:
        shutil.rmtree(out, ignore_errors=True)
    else:
        lines.append("outputs kept in " + out)
    return section("design", ok, *lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gpu", action="store_true", help="tree and pins only (the build runs this: no GPU in a build container)")
    ap.add_argument("--design", action="store_true", help="also design 3HTN both ways and compare the bytes (a few minutes)")
    ap.add_argument("--gcds", default="", help="the GCD indices to test (default: all visible)")
    ap.add_argument("--keep", default=None, help="--design: keep its outputs in this directory")
    a = ap.parse_args()

    say("proteinmpnn check -- %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    check_tree()
    check_pins()
    if not a.no_gpu:
        check_devices(a.gcds)
        check_libraries()
        check_draw()
        if a.design:
            check_design(a.keep)

    bad = [n for n, ok in RESULTS if not ok]
    say("\n%s: %s" % ("CHECK FAILED" if bad else "CHECK PASS",
                      ", ".join("%s %s" % (n, "pass" if ok else "FAIL") for n, ok in RESULTS)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
