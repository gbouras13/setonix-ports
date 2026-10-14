# tests

- `test_*.py`: the CPU tests of the port's eight opt-in fixes, as the native port runs them (`slurm/cpucheck.sbatch` there). Each
  checks a wrapper against the stock statement it wraps, and its hook installing in fresh interpreters. `opendde check --no-gpu
  --patch-tests` runs all eight, and the image build runs that command. `test_patch_hook.py` runs in its v2 form
  (`ODDE_HOOK_VERSION=2`); the v1 mask patch is not in the image.
- `q600.json`, `q4096.json`: the synthetic queries the native port measured with (one chain of 600 residues; four chains of 1,024).
- `det600_stock.cif`: the native port's 600-token prediction on one GCD under `--det 1`, stock line (md5
  `bb6788dbfd2030283a3ba2adb8773155`; pLDDT 77.45848083496094). `--det 1` is fully deterministic on this card, and the patched line
  reproduces it byte for byte natively. `opendde check --predict` requires both lines to reproduce it inside the image.
