#!/usr/bin/env python3
"""repin163.py KIT_COLABFOLD_DIR KITPIN_DIR -- the kit's stock pin restated for the Setonix port's stock: ColabFold 1.6.3 with
alphafold-colabfold 2.3.20 (the user's choice, 2026-10-05) on jax 0.10.2 + jax-rocm7 0.10.2, instead of the kit's 1.6.1 / 2.3.13 on
jax 0.5.3 + CUDA 12.

Builds KITPIN_DIR, a tree the kit reads through its own documented variable MODEL_OPT ("where the package finds the tree and the pins"):
  run.sh                        the kit's run.sh, byte for byte (its HERE is then this tree: its pin check reads the pins below)
  stock/check_pins.py           the kit's, byte for byte
  stock/PINS.json               the kit's PINS.json with the upstream versions, wheels, declared ranges, package list, image environment and
                                stack restated for 1.6.3 / ROCm (field "setonix_repin" records what changed and the source pin's sha256);
                                the weights block (the same five AlphaFold2-Multimer v3 files) and the three pinned file paths are unchanged
  stock/src/<three files>       the INSTALLED 1.6.3 / 2.3.20 copies of the three files the kit pins by bytes (alphafold/model/modules.py,
                                colabfold/batch.py, colabfold/input.py)
  stock/<two wheels>            the PyPI wheels of colabfold 1.6.3 and alphafold-colabfold 2.3.20 (the kit's settings.py reads
                                colabfold_batch's option table from the colabfold wheel); must already be in STOCK_WHEELS_DIR
  environment/requirements.lock this venv's `pip freeze --all` (less the two editable kit packages): the stack the package check reads
  configs/                      left to the caller (scripts/make-kitpin.sh writes configs/mi250x.env)
  tests -> the kit's tests/     (the warm verb's public input)
Nothing of the kit tree is modified. Runs on the venv's interpreter, inside a job (not on a login node).
"""
import hashlib
import importlib.metadata as md
import json
import os
import shutil
import subprocess
import sys

STOCK_FILES = ("alphafold/model/modules.py", "colabfold/batch.py", "colabfold/input.py")
# CFS_REPIN_IMAGE: set by the container build (ports/colabfold/container/Dockerfile) to a one-line description of the image's stack, so
# the pins describe that stack instead of the native one (Cray python, Pawsey's ROCm module)
IMAGE = os.environ.get("CFS_REPIN_IMAGE", "").strip()
CHECK_PACKAGES = ["colabfold", "alphafold-colabfold", "jax", "jaxlib", "jax-rocm7-plugin", "jax-rocm7-pjrt", "dm-haiku", "dm-tree", "numpy",
                  "scipy", "biopython", "ml_collections", "chex", "pandas"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dist_file(dist, rel):
    p = md.distribution(dist).locate_file(rel)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"{dist}: {rel} not installed")
    return str(p)


def main(argv):
    kit, out = os.path.abspath(argv[0]), os.path.abspath(argv[1])
    wheels = os.environ.get("STOCK_WHEELS_DIR", os.path.join(out, "stock"))
    src_pins = os.path.join(kit, "stock", "PINS.json")
    with open(src_pins, encoding="utf-8") as f:
        pins = json.load(f)
    os.makedirs(os.path.join(out, "stock", "src"), exist_ok=True)
    os.makedirs(os.path.join(out, "environment"), exist_ok=True)
    os.makedirs(os.path.join(out, "configs"), exist_ok=True)
    shutil.copy2(os.path.join(kit, "run.sh"), os.path.join(out, "run.sh"))
    shutil.copy2(os.path.join(kit, "stock", "check_pins.py"), os.path.join(out, "stock", "check_pins.py"))
    if not os.path.lexists(os.path.join(out, "tests")):
        os.symlink(os.path.join(kit, "tests"), os.path.join(out, "tests"))
    for rel in STOCK_FILES:
        dist = "colabfold" if rel.startswith("colabfold/") else "alphafold-colabfold"
        dst = os.path.join(out, "stock", "src", rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(dist_file(dist, rel), dst)

    cf_whl = "colabfold-1.6.3-py3-none-any.whl"
    af_whl = "alphafold_colabfold-2.3.20-py3-none-any.whl"
    for w in (cf_whl, af_whl):
        p = os.path.join(wheels, w)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{p}: fetch it first (pip download --no-deps, on the login node)")
        if os.path.abspath(wheels) != os.path.join(out, "stock"):
            shutil.copy2(p, os.path.join(out, "stock", w))

    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], capture_output=True, text=True, check=True).stdout.splitlines()
    keep = [ln for ln in freeze if ln.strip() and not ln.startswith("-e ") and " @ " not in ln and not ln.lower().startswith(("colabfold-opt", "colabfold_opt", "opt-core", "opt_core"))]
    lock = os.path.join(out, "environment", "requirements.lock")
    with open(lock, "w", encoding="utf-8") as f:
        f.write(f"# Setonix port stack (repin163.py): `pip freeze --all` of the venv {sys.prefix}, less the two\n"
                "# editable kit packages (colabfold_opt, opt_core). "
                + (f"{IMAGE}\n" if IMAGE else
                   "Python 3.12.12 (Cray), jax 0.10.2 + jax-rocm7-plugin/pjrt 0.10.2 against the\n"
                   "# rocm/7.2.4 module (module use /software/setonix/unsupported); ColabFold 1.6.3 + alphafold-colabfold 2.3.20 from PyPI.\n"))
        f.write("\n".join(keep) + "\n")

    meta = md.metadata("colabfold")
    ranges = [r for r in (meta.get_all("Requires-Dist") or [])
              if "extra ==" not in r or 'extra == "alphafold-minus-jax"' in r or 'extra == "alphafold"' in r]
    up = pins["upstream"]
    up["colabfold"].update(
        version=md.version("colabfold"),
        install="pip install colabfold[alphafold-minus-jax]==1.6.3 jax==0.10.2 jaxlib==0.10.2 jax-rocm7-plugin==0.10.2 jax-rocm7-pjrt==0.10.2 "
                "(scripts/setup-venv.sh; == the PyPI wheel stock/" + cf_whl + ")",
        wheel={"file": "stock/" + cf_whl, "bytes": os.path.getsize(os.path.join(out, "stock", cf_whl)),
               "sha256": sha256(os.path.join(out, "stock", cf_whl))},
        declared_ranges=sorted(set(r.split(";")[0].strip() for r in ranges)),
        note="the wheel's METADATA Requires-Dist (base + alphafold extras); jax itself is the ROCm build, outside the extras")
    up["alphafold_colabfold"].update(
        version=md.version("alphafold-colabfold"),
        install="pip install alphafold-colabfold==2.3.20 (pulled by colabfold 1.6.3; == the PyPI wheel stock/" + af_whl + ")",
        wheel={"file": "stock/" + af_whl, "bytes": os.path.getsize(os.path.join(out, "stock", af_whl)),
               "sha256": sha256(os.path.join(out, "stock", af_whl))},
        pin={"note": "PyPI wheel bytes (sha256 above)", "source": "pypi", "package": "alphafold-colabfold", "version": md.version("alphafold-colabfold")})
    env_now = {k: os.environ.get(k) for k in ("JAX_PLATFORMS", "XLA_PYTHON_CLIENT_MEM_FRACTION", "TF_FORCE_UNIFIED_MEMORY", "PYTHONHASHSEED")}
    pins["image"] = {
        "tested_on": "Setonix GPU nodes: AMD Instinct MI250X (gfx90a), 8 GCDs x 64 GB, SLES, module rocm/7.2.4 -- documentation only",
        "python": sys.executable, "python_version": sys.version.split()[0],
        "colabfold_batch": os.path.join(os.path.dirname(sys.executable), "colabfold_batch"),
        "base": IMAGE or "bare-metal venv on Cray python 3.12.12 (no container)",
        "env": {"JAX_PLATFORMS": "rocm,cpu", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95", "TF_FORCE_UNIFIED_MEMORY": "0", "PYTHONHASHSEED": "0"},
        "env_note": "scripts/cf-env.sh. XLA_CLIENT_MEM_FRACTION must stay UNSET: jax 0.10.2 raises inside the ROCm plugin's initialize() when "
                    "both fraction names are set, and jax then runs on the CPU (af3_jax port); colabfold/batch.py sets "
                    "XLA_PYTHON_CLIENT_MEM_FRACTION=4.0 when it is unset, so the port always sets that name. Values at re-pin time: %s" % env_now,
        "gpu": "AMD Instinct MI250X GCD (gfx90a), 64 GB HBM2e",
    }
    pins["check_packages"] = CHECK_PACKAGES
    pins["freeze"] = {"file": "environment/requirements.lock", "packages": len(keep)}
    jaxv = md.version("jax")
    pins["stack"] = {"key_form": pins["stack"].get("key_form"), "kernel_key_form": pins["stack"].get("kernel_key_form"),
                     "jax": jaxv, "jaxlib": md.version("jaxlib"), "jax_rocm7_plugin": md.version("jax-rocm7-plugin"),
                     "rocm": "7.2.4 (AMD's apt packages, in the image)" if IMAGE else "7.2.4 (Pawsey unsupported module tree)",
                     "python": sys.version.split()[0], "dm_haiku": md.version("dm-haiku"),
                     "gpu": "AMD Instinct MI250X (gfx90a)", "compute_cap": "gfx90a (the port's synthetic provider column: 8.1)",
                     "note": "no CUDA: the kit's stack key reads cu<unknown>-sm81 (nvidia-smi shim, CFS_SHIM_CC)"}
    pins["hardware"] = {"gate": "the kit's gate_gpu reads nvidia-smi: on Setonix the port's shim (scripts/bin/nvidia-smi) answers from sysfs"}
    pins["setonix_repin"] = {
        "date": "2026-10-05", "tool": "af3-setonix/colabfold/tools/repin163.py",
        "source_pins": src_pins, "source_pins_sha256": sha256(src_pins),
        "changed": ["upstream.colabfold", "upstream.alphafold_colabfold", "image", "check_packages", "freeze", "stack", "hardware"],
        "unchanged": ["weights", "stock_files", "stock_proof", "model"],
        "stock_files_sha256": {rel: sha256(os.path.join(out, "stock", "src", rel)) for rel in STOCK_FILES},
        "why": "the user chose ColabFold 1.6.3 (latest release) as stock for the Setonix port; jax 0.10.2 is the newest ROCm plugin that loads "
               "on Setonix (0.11.2 SIGBUSes: af3_jax port B1) and is inside 1.6.3's declared range (jax>=0.6.2,<0.12)"}
    with open(os.path.join(out, "stock", "PINS.json"), "w", encoding="utf-8") as f:
        json.dump(pins, f, indent=1)
        f.write("\n")
    print("REPIN_OK colabfold %s alphafold-colabfold %s jax %s -> %s (lock: %d packages)" % (
        up["colabfold"]["version"], up["alphafold_colabfold"]["version"], jaxv, out, len(keep)))


if __name__ == "__main__":
    main(sys.argv[1:])
