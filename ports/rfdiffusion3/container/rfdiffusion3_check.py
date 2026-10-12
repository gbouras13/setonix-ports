#!/usr/bin/env python
"""rfdiffusion3_check.py -- is this image whole, is it running on its own libraries, and are the two gfx90a fixes in force?

    rfdiffusion3 check [--no-gpu] [--design] [--ckpt FILE]

Sections, each PASS or FAIL, and a verdict line at the end:
  tree        both interpreters, torch, the kit's commit, upstream's archive, apex's provenance, and the two fixes this port depends on
  pins        the kit's own stock/check_pins.py in BOTH interpreters -- `pristine` for venv-stock, `not pristine` for venv-kit, which is
              what the ten-file overlay being installed looks like -- and the overlay's own `install status`
  libraries   every loaded GPU library comes from inside the image. Singularity's --rocm binds the HOST's ROCm 6.3.0 over the image's
              7.2 (same sonames) and that is a silent wrong answer, so it is a failure here, not a warning
  devices     the GCDs torch sees, their architecture, and a matmul on each against a CPU reference
  rmsnorm     THE check for this card: apex's FusedRMSNorm -- which upstream binds as every RMSNorm of the network -- against a float64
              reference. The sources the kit pins assume a 32-lane warp and are 29 % wrong on a 64-lane wavefront, which makes every
              design rubbish; the image builds apex with the width passed explicitly, and this proves which one is installed
  weights     whether a checkpoint is mounted where RFD3_CKPT points, and whether it has the digest the kit's PINS.json records
  design      (--design) a 2-design run through `off` and through `exact`, compared design by design: on this card they are identical

Exit 0 when every section selected passes, 1 otherwise. Writes nothing outside RFD_LOCAL.
"""
import argparse
import importlib.util
import os
import re
import subprocess
import sys
import tempfile

W = os.environ.get("RFD_ROOT", "/opt/rfdiffusion3")
KIT = os.path.join(W, "kit", "rfdiffusion3")
VENVS = {"venv-stock": "pristine", "venv-kit": "any"}
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
             read(os.path.join(W, "upstream.txt"), "upstream unknown"),
             read(os.path.join(W, "apex.txt"), "apex unknown")]
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
    for v in VENVS:
        p = os.path.join(W, v, "bin", "python")
        lines.append("%-11s %s" % (v, p if os.path.exists(p) else "ABSENT"))
        ok = ok and os.path.exists(p)
    for want in ("opt_core", "rfdiffusion3_opt", "rfd3", "foundry"):
        spec = importlib.util.find_spec(want)
        origin = getattr(spec, "origin", None)
        lines.append("%-17s %s" % (want, origin or "ABSENT"))
        if not origin or not os.path.realpath(origin).startswith(os.path.realpath(W) + os.sep):
            ok = False
    # the two fixes, and the setting
    boot = os.environ.get("RFD_BOOT", "")
    cap = os.environ.get("DEBUG_CLR_GRAPH_PACKET_CAPTURE", "UNSET -- rfd-env.sh was not sourced")
    lines.append("fix 1  apex built with the 64-lane wavefront patch (see apex line above; the rmsnorm section proves it)")
    lines.append("fix 2  RFD_BOOT = %s  (the fused transition steps aside on this card; the kit's APPLIED line then reads fused_calls=0)"
                 % (boot or "EMPTY -- exact will NOT be stock's designs here"))
    lines.append("setting DEBUG_CLR_GRAPH_PACKET_CAPTURE = %s" % cap)
    if boot != "fzt_rows_off":
        lines.append("RFD_BOOT should be fzt_rows_off on gfx90a: without it `exact` diverges from stock (one design 20.2 A away)")
        ok = False
    if os.environ.get("DEBUG_CLR_GRAPH_PACKET_CAPTURE") != "0":
        lines.append("DEBUG_CLR_GRAPH_PACKET_CAPTURE must be 0 on ROCm 7.2: a replayed graph corrupts the host heap")
        ok = False
    for f in (os.path.join(KIT, "run.sh"), os.path.join(W, "foundry-4010e3e2", "pyproject.toml"),
              os.path.join(W, "scripts", "rfd-env.sh"), os.path.join(W, "scripts", "boot", "sitecustomize.py")):
        if not os.path.exists(f):
            lines.append("MISSING %s" % f)
            ok = False
    return section("tree", ok, *lines)


# ---------------------------------------------------------------------------------------------------------------- pins
def check_pins():
    ok, lines = True, []
    for v, expect in VENVS.items():
        py = os.path.join(W, v, "bin", "python")
        p = subprocess.run([py, "-I", os.path.join(KIT, "stock", "check_pins.py"), "--expect", expect],
                           capture_output=True, text=True, timeout=300)
        out = (p.stdout + p.stderr).strip().splitlines()
        lines.append("%-11s --expect %-9s rc=%d" % (v, expect, p.returncode))
        for ln in out:
            lines.append("    " + ln)
        ok = ok and p.returncode == 0
    env = dict(os.environ, MODEL_OPT=KIT)
    p = subprocess.run([os.path.join(W, "venv-kit", "bin", "python"), "-m", "rfdiffusion3_opt.install", "status"],
                       capture_output=True, text=True, timeout=300, env=env)
    head = (p.stdout + p.stderr).strip().splitlines()
    lines.append("overlay status rc=%d: %s" % (p.returncode, head[0] if head else "(no output)"))
    ok = ok and p.returncode == 0
    return section("pins", ok, *lines)


# ---------------------------------------------------------------------------------------------------------------- libraries
GPU_LIB = re.compile(r"lib(amdhip|hsa-runtime|rocblas|hipblaslt|MIOpen|rocrand|hiprand|rccl|amd_comgr|rocsolver|rocsparse|magma|"
                     r"torch_hip|aotriton|hiprtc|hipfft|rocfft|hipsolver|hipsparse|roctracer|rocprofiler)")


def check_libraries():
    """Every mapped GPU library must live inside the image. Singularity --rocm binds the host's ROCm 6.3.0 with the same sonames."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.zeros(8, device="cuda").sum().item()
    except Exception:                                                        # noqa: BLE001
        pass
    inside = tuple(os.path.realpath(p) + os.sep for p in (W, "/opt/rocm", "/opt/python", "/usr/lib", "/lib"))
    loaded, outside = [], []
    for line in read("/proc/self/maps").splitlines():
        parts = line.split()
        if len(parts) < 6 or not parts[-1].startswith("/"):
            continue
        path = parts[-1]
        base = os.path.basename(path)
        if not GPU_LIB.match(base) or path in loaded:
            continue
        loaded.append(path)
        if not os.path.realpath(path).startswith(inside):
            outside.append(path)
    lines = ["%d GPU libraries mapped, all from inside the image" % len(loaded)] if not outside else \
            ["%d of %d GPU libraries come from OUTSIDE the image (run without --rocm):" % (len(outside), len(loaded))] + outside
    return section("libraries", not outside, *lines)


# ---------------------------------------------------------------------------------------------------------------- devices
def check_devices():
    import torch
    if not torch.cuda.is_available():
        return section("devices", False, "torch sees no GPU (ROCR_VISIBLE_DEVICES=%s)" % os.environ.get("ROCR_VISIBLE_DEVICES", "unset"))
    ok, lines = True, []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        x = torch.randn(512, 512, device="cuda:%d" % i)
        got = (x @ x).sum().item()
        want = (x.cpu() @ x.cpu()).sum().item()
        close = abs(got - want) <= 1e-2 * max(1.0, abs(want))
        ok = ok and close
        lines.append("GCD %d  %-24s %-8s %6.1f GiB  matmul %s"
                     % (i, p.name, getattr(p, "gcnArchName", "?"), p.total_memory / 2**30, "ok" if close else "WRONG"))
    return section("devices", ok, *lines)


# ---------------------------------------------------------------------------------------------------------------- rmsnorm
def check_rmsnorm():
    """apex's FusedRMSNorm against a float64 reference of upstream's own rule, in fp32, at shapes with several rows per block.

    The unpatched kernel is ~29 % wrong here and every design it makes is rubbish; the patched one is torch.nn.RMSNorm's own error.
    One row per block cannot tell them apart, so the shapes below all have many.
    """
    import torch
    if not torch.cuda.is_available():
        return section("rmsnorm", False, "needs a GPU")
    try:
        from apex.normalization.fused_layer_norm import FusedRMSNorm
    except Exception as e:                                                   # noqa: BLE001
        return section("rmsnorm", False, "apex's FusedRMSNorm does not import: %r" % (e,),
                       "a kit mode refuses to run without it (KERNELS rmsnorm=fallback, exit 5)")
    eps, ok, lines, worst = 1e-5, True, [], 0.0
    torch.manual_seed(0)
    for rows, cols in ((4, 128), (64, 256), (1024, 512), (256, 1024)):
        w = torch.randn(cols, device="cuda").abs() + 0.5
        m = FusedRMSNorm(cols, eps=eps).cuda()
        with torch.no_grad():
            m.weight.copy_(w)
            x = torch.randn(rows, cols, device="cuda")
            got = m(x).double().cpu()
        xd, wd = x.double().cpu(), w.double().cpu()
        ref = xd * torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + eps) * wd
        rel = ((got - ref).abs() / ref.abs().clamp_min(1e-12)).max().item()
        worst = max(worst, rel)
        good = rel < 1e-5
        ok = ok and good
        lines.append("rows %-5d cols %-5d max relative error %.3e %s" % (rows, cols, rel, "" if good else "<<< WRONG"))
    lines.append("worst %.3e against float64 (torch.nn.RMSNorm's own error on this card is ~2e-7)" % worst)
    if not ok:
        lines.append("this is the UNPATCHED kernel: it assumes a 32-lane warp, and on a 64-lane wavefront the block's rows read")
        lines.append("each other's partial sums. No design made with it is a protein. Rebuild the image.")
    return section("rmsnorm", ok, *lines)


# ---------------------------------------------------------------------------------------------------------------- weights
def check_weights(ckpt):
    import json
    pins = json.load(open(os.path.join(KIT, "stock", "PINS.json")))["checkpoint"]
    lines = ["RFD3_CKPT = %s" % ckpt, "the image ships no checkpoint: mount it (-B /path:/weights) or `rfdiffusion3 fetch DIR`"]
    if not os.path.isfile(ckpt):
        lines.append("not present -- `design` and `warm` need it; every other section here does not")
        return section("weights", True, *lines)                              # absent is not a failure of the image
    size = os.path.getsize(ckpt)
    lines.append("%d bytes (the pin records %s)" % (size, pins.get("bytes")))
    ok = size == pins.get("bytes")
    if not ok:
        lines.append("size differs from the pin: this is not the checkpoint the kit records")
    return section("weights", ok, *lines)


# ---------------------------------------------------------------------------------------------------------------- design
def check_design(ckpt):
    """Two designs through `off` and through `exact`, compared. On this card, with both fixes in, they are identical."""
    if not os.path.isfile(ckpt):
        return section("design", False, "no checkpoint at %s" % ckpt)
    spec = os.path.join(W, "foundry-4010e3e2", "models", "rfd3", "docs", "examples", "protein_binder_design.json")
    out = tempfile.mkdtemp(prefix="rfd-check-", dir=os.environ.get("RFD_LOCAL", "/tmp"))
    lines, dirs = [], {}
    for mode in ("off", "exact"):
        d = os.path.join(out, mode)
        p = subprocess.run(["/usr/local/bin/rfdiffusion3", "design", "--mode", mode, "--det", "1",
                            "inputs=%s" % spec, "out_dir=%s" % d, "n_batches=1",
                            "diffusion_batch_size=1", "seed=101", "ckpt_path=%s" % ckpt],
                           capture_output=True, text=True, timeout=5400)
        n = len([f for f in os.listdir(d)] if os.path.isdir(d) else [])
        lines.append("%-5s rc=%d, %d files" % (mode, p.returncode, n))
        if p.returncode != 0:
            lines += ["    " + ln for ln in (p.stdout + p.stderr).strip().splitlines()[-6:]]
            return section("design", False, *lines)
        dirs[mode] = d
    c = subprocess.run(["/usr/local/bin/rfdiffusion3", "compare", dirs["off"], dirs["exact"], "--quiet"],
                       capture_output=True, text=True, timeout=600)
    tail = (c.stdout + c.stderr).strip().splitlines()
    lines += ["    " + ln for ln in tail[-4:]]
    ok = c.returncode == 0
    if not ok:
        lines.append("exact did not reproduce stock here -- check that RFD_BOOT=fzt_rows_off is in force (tree section)")
    return section("design", ok, *lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gpu", action="store_true", help="skip everything that needs a GCD (the build runs this)")
    ap.add_argument("--design", action="store_true", help="also design through both routes and compare (needs the checkpoint)")
    ap.add_argument("--ckpt", default=os.environ.get("RFD3_CKPT", "/weights/rfd3/rfd3_latest.ckpt"))
    a = ap.parse_args(argv)
    say("rfdiffusion3 check  (%s)" % W)
    check_tree()
    check_pins()
    if not a.no_gpu:
        check_libraries()
        check_devices()
        check_rmsnorm()
    check_weights(a.ckpt)
    if a.design and not a.no_gpu:
        check_design(a.ckpt)
    bad = [n for n, ok in RESULTS if not ok]
    say("\n%s  (%d sections, %d failed%s)" % ("ALL PASS" if not bad else "FAILED", len(RESULTS), len(bad),
                                              "" if not bad else ": " + ", ".join(bad)))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
