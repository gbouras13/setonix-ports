#!/bin/sh
# build_venvs.sh -- the image's two interpreters, built in one layer (container/Dockerfile step 5).
#
# This kit is a FILE OVERLAY on the installed `rfd3` package, so a stock run and a kit run can never share a site-packages:
#   venv-stock   pristine. `--mode off` runs there, under `python -I`.
#   venv-kit     the same environment, plus the kit's ten-file rfd3/model overlay. `exact` and `fast` run there.
# Both get the same packages at the versions the native port validated (--no-deps, so the lock IS the environment), upstream
# from the kit's own source archive, the apex wheel built in the previous step, and the kit's two packages editable -- the
# kit's core gate reads [tool.opt_core] from above the package, so a wheel install of them is refused by name.
#
# Then: the overlay into venv-kit only; the kit's own pin check in each (`pristine` for venv-stock, `not pristine` for
# venv-kit is what a correct install looks like); the gfx90a prune; and jdupes, which hardlinks the two trees together --
# they differ only in the ten overlaid files, so the image carries one copy of everything else.
set -eu
W=${W:-/opt/rfdiffusion3}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/rocm7.2}
FOUNDRY_VERSION=${FOUNDRY_VERSION:-0.2.1.dev13+g4010e3e2e}
PRUNE_GPU_ARCH=${PRUNE_GPU_ARCH:-gfx90a}
APEX_WHEEL_DIR=${APEX_WHEEL_DIR:-/tmp/apexwheel}
VENVS="venv-stock venv-kit"

for v in $VENVS; do
  echo "== $v: environment"
  /opt/python/bin/python3.12 -m venv "$W/$v"
  "$W/$v/bin/python" -m pip install --quiet --no-compile --upgrade "setuptools>=64" wheel
  "$W/$v/bin/python" -m pip install --quiet --no-deps --no-compile \
      --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple \
      -r "$W/environment/requirements.lock"
  SETUPTOOLS_SCM_PRETEND_VERSION="$FOUNDRY_VERSION" \
      "$W/$v/bin/python" -m pip install --quiet --no-deps --no-compile "$W/foundry-4010e3e2[rfd3]"
  "$W/$v/bin/python" -m pip install --quiet --no-compile --no-deps "$APEX_WHEEL_DIR"/apex-*.whl
  "$W/$v/bin/python" -m pip check
  # the kit's own install verb: its two packages editable, then its pin check (--expect any at this point: the overlay
  # is not in yet for venv-kit, and never will be for venv-stock)
  PATH="$W/$v/bin:$PATH" sh "$W/kit/rfdiffusion3/run.sh" install
  "$W/$v/bin/python" -m pip freeze --all > "$W/environment/image-freeze-$v.txt"
done

echo "== the overlay, into venv-kit only"
PATH="$W/venv-kit/bin:$PATH" MODEL_OPT="$W/kit/rfdiffusion3" \
    "$W/venv-kit/bin/python" -m rfdiffusion3_opt.install apply

echo "== the kit's pin check in each interpreter"
: > "$W/pins.txt"
for v in $VENVS; do
  expect=any
  [ "$v" = venv-stock ] && expect=pristine
  echo "== $v --expect $expect" >> "$W/pins.txt"
  "$W/$v/bin/python" -I "$W/kit/rfdiffusion3/stock/check_pins.py" --expect "$expect" >> "$W/pins.txt" 2>&1
done
cat "$W/pins.txt"

echo "== prune the ROCm kernels of other cards, and hardlink the two trees"
: > "$W/rocm-pruned.txt"
for v in $VENVS; do
  L=$("$W/$v/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
  python3 "$W/container/prune_rocm_arch.py" --keep "$PRUNE_GPU_ARCH" \
      "$L/torch/lib/hipblaslt" "$L/torch/lib/rocblas" "$L/torch/lib/aotriton.images" "$L/torch/share/miopen" \
      >> "$W/rocm-pruned.txt"
done
# jdupes exits non-zero when it finds nothing to do; the sizes below are the real check.
jdupes --link-hard --recurse --quiet "$W/venv-stock" "$W/venv-kit" >> "$W/rocm-pruned.txt" 2>&1 || true
du -sh "$W/venv-stock" "$W/venv-kit" | tee -a "$W/rocm-pruned.txt"
du -sh --apparent-size "$W/venv-stock" "$W/venv-kit" >> "$W/rocm-pruned.txt" 2>/dev/null || true
