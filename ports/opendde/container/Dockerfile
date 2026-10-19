# syntax=docker/dockerfile:1.7
# OpenDDE 1.1.1 for Pawsey Setonix (AMD Instinct MI250X, gfx90a), on the optimisation kit's `big` line: one GCD, or row-sharded over up to
# 8 by the kit's own launcher, with the port's eight opt-in fixes. The image IS the native port's tree, at /opt/opendde:
#   venv/     torch 2.14.0+rocm7.2 + triton-rocm 3.8.0 (download.pytorch.org/whl/rocm7.2) and OpenDDE's stack, pinned exactly by
#             environment/requirements.lock (the versions the native process resolved) and installed --no-deps; opendde 1.1.1 from the
#             kit's own wheel; the kit's two packages (opt_core, opendde_opt) editable, as natively;
#   kit/      the optimisation kit at its pinned commit (opendde/, common/);
#   patches/  the eight opt-in fixes and the hook chain that installs them (hook7 -> ... -> hook2), each through the kit's own
#             opt_core.autoload.patch_attr_at_import: the installed packages are never edited;
#   scripts/  the port's environment, its prediction script (the native predict.sh and predict_patched7.sh as one script) and the
#             per-rank cache hook; tools/ the structure comparison; tests/ the patches' CPU tests, the 600- and 4,096-token queries and
#             the native port's 600-token --det 1 reference structure.
# NO WEIGHTS: like every image here but proteinmpnn's, it reads them from outside. `opendde fetch-weights DIR` runs the kit's own downloader
# (upstream's URL table: the checkpoint from Hugging Face aurekaresearch/OpenDDE, and four common/ files), each checked against the kit's
# pinned sha256.
#
# The gfx90a recipe (af3-setonix/OPENDDE_PORT_LOG.md), none of it an edit to a kit or OpenDDE file:
#   - upstream's torch triangle kernels by stated flag (no cuEquivariance on ROCm), the sharded triangle-attention core as torch
#     statements, and the four levers whose kernels cannot run on gfx90a (the apb sampler kernels request tf32) composed out by name;
#   - HIP_LAUNCH_BLOCKING=1 (with 0 a rank segfaults in the diffusion roll-out), LAYERNORM_TYPE=torch (the fused LayerNorm needs nvcc),
#     and the kit's sub-64-GiB size gates (its card probe reads nvidia-smi only);
#   - the patched line (the default): v2 dense-window construction (byte-identical), explicit triangle-attention blocks,
#     ROWPAIR_TRIMUL_SUB=128, the native allocator, the A-block threshold, ROWPAIR_TRIATT_ROWBLOCK=256, query-blocked stock attention
#     and row-blocked APB pair bias in the structural refiner. Measured: 2.45x stock at 4,096 tokens on 8 GCDs with the structure inside
#     run-to-run noise; complete predictions to 9,216 tokens on one node.
#
# Run on Setonix WITHOUT --rocm (it binds the host's ROCm 6.3.0 libraries, same sonames). `opendde check` fails if any GPU library was
# loaded from outside the image. Images cannot be built on Setonix; GitHub Actions builds this one (.github/workflows/opendde-image.yml).
# Context: ports/opendde/.

ARG BASE_IMAGE=rocm/dev-ubuntu-24.04:7.2.4@sha256:bdc8e61026cbb844ede93d44d2c50055f51ebb2041906b60182bf3bee3139054
FROM ${BASE_IMAGE}
# Each build argument is declared just before the step that uses it: every RUN after an ARG sees it in its environment, so a changed
# value (SOURCE_REV changes with every commit) re-runs only the steps from its declaration on, and the CI build's layer cache holds.
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# ------------------------------------------------------------------------------------------------ 1. tools
# No ROCm math libraries are installed: the torch 2.14.0+rocm7.2 wheel carries its own complete ROCm 7.2 runtime in torch/lib, and that
# is the one the native port runs on too. git and curl fetch the kit and the interpreter (and, at run time, fetch-weights the weights);
# python3 runs the prune script. Every package of the lock has a cp312 wheel, so no compiler is installed.
RUN mkdir -p /opt/opendde \
 && apt-get update --quiet \
 && apt-get install --yes --quiet --no-install-recommends git curl ca-certificates python3 \
 && rm -rf /var/lib/apt/lists/* \
 && dpkg-query -W -f '${Package} ${Version}\n' rocm-core hip-runtime-amd hsa-rocr comgr \
      > /opt/opendde/rocm-packages.txt

# ------------------------------------------------------------------------------------------------ 2. Python 3.12
# The native venv's base is the OpenFold3 conda env's CPython 3.12, a /scratch tree with no equivalent in an image. This is the same
# minor version from python-build-standalone (the build the colabfold and boltz2 images use); x86_64_v3 covers Setonix's Trento (Zen 3).
# Nothing in the port's numerics depends on the interpreter's patch level: `opendde check --predict` compares the bytes of a prediction
# with the native port's.
ARG PYTHON_URL=https://github.com/astral-sh/python-build-standalone/releases/download/20260211/cpython-3.12.12%2B20260211-x86_64_v3-unknown-linux-gnu-install_only.tar.gz
ARG PYTHON_SHA256=e10e333796fab1a7793171ea5c2b8cadc6f9c5dfdcc06815fa402c05cbd76d81
RUN curl -fsSL -o /tmp/python.tar.gz "$PYTHON_URL" \
 && echo "$PYTHON_SHA256  /tmp/python.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/python.tar.gz -C /tmp && mv /tmp/python /opt/python && rm -f /tmp/python.tar.gz \
 && /opt/python/bin/python3.12 -V

# ------------------------------------------------------------------------------------------------ 3. the kit, at its pin (sparse)
ARG KIT_REPO=https://github.com/anthropics/uplifting-biomolecular-modeling.git
ARG KIT_COMMIT=f4f62fa6592ae4938d49b1757bea0cfeff9f468e
RUN git clone --quiet --filter=blob:none --no-checkout "$KIT_REPO" /opt/opendde/kit \
 && git -C /opt/opendde/kit sparse-checkout set --cone opendde common \
 && git -C /opt/opendde/kit fetch --quiet --filter=blob:none origin "$KIT_COMMIT" \
 && git -C /opt/opendde/kit -c advice.detachedHead=false checkout --quiet "$KIT_COMMIT" \
 && test "$(git -C /opt/opendde/kit rev-parse HEAD)" = "$KIT_COMMIT" \
 && test -z "$(git -C /opt/opendde/kit status --porcelain --untracked-files=no)" \
 && rm -rf /opt/opendde/kit/.git \
 && echo "kit $KIT_REPO @ $KIT_COMMIT" > /opt/opendde/kit.txt

# ------------------------------------------------------------------------------------------------ 4. the venv, pinned, pruned to gfx90a
# setuptools and wheel first, at the native venv's versions; then the lock's wheels (--no-deps, wheels only); opendde 1.1.1 from the
# kit's wheel (--no-deps: its torch==2.7.1 pin and its CUDA "gpu" extra must not be honoured); the kit's two packages editable
# (--no-deps: opendde_opt declares cuda-bindings, CUDA only, and ninja, for upstream's nvcc LayerNorm), as natively. `pip check` is
# filtered: opendde pins torch 2.7.1, numpy 2.4.1 and rdkit 2025.9.3 and runs, as natively, on 2.14.0+rocm7.2, 2.5.3 and 2025.9.6;
# opendde_opt's two CUDA-side requirements are absent on purpose. Those five lines are expected; anything else fails the build.
# The kit's own check_pins then states opendde is installed at its pin, file for file.
#
# The prune: of the ~14 GiB in torch/, ~6 GiB are GPU kernels for cards that are not this one (hipBLASLt, rocBLAS, aotriton, MIOpen's
# databases); it runs in the same layer as the install, so the layer never holds the unpruned tree. The large .so files stay: they are
# DT_NEEDED entries of libtorch_hip.so.
ARG PRUNE_GPU_ARCH=gfx90a
ARG TORCH_INDEX=https://download.pytorch.org/whl/rocm7.2
COPY container/prune_rocm_arch.py /opt/opendde/container/
COPY environment/requirements.lock /opt/opendde/environment/
RUN /opt/python/bin/python3.12 -m venv /opt/opendde/venv \
 && P=/opt/opendde/venv/bin/python \
 && $P -m pip install --quiet --no-deps --no-compile setuptools==84.0.0 wheel==0.48.0 \
 && $P -m pip install --quiet --no-deps --no-compile --only-binary :all: \
      --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple -r /opt/opendde/environment/requirements.lock \
 && $P -m pip install --quiet --no-deps --no-compile /opt/opendde/kit/opendde/stock/opendde-1.1.1-py3-none-any.whl \
 && $P -m pip install --quiet --no-deps --no-compile --no-build-isolation \
      -e /opt/opendde/kit/common/opt_core -e /opt/opendde/kit/opendde/opt \
 && { $P -m pip check > /tmp/pipcheck.txt 2>&1 || true; } \
 && grep -v -E "^opendde 1\.1\.1 has requirement (torch==2\.7\.1|numpy==2\.4\.1|rdkit==2025\.9\.3), but you have |^opendde[-_]opt 0\.2\.72 requires (cuda-bindings|ninja), which is not installed" /tmp/pipcheck.txt \
      | grep -v -i -E "^No broken requirements found" > /tmp/pipcheck.unexpected || true \
 && cat /tmp/pipcheck.txt \
 && { [ ! -s /tmp/pipcheck.unexpected ] || { echo "UNEXPECTED pip check lines:"; cat /tmp/pipcheck.unexpected; exit 1; }; } \
 && rm -f /tmp/pipcheck.txt /tmp/pipcheck.unexpected \
 && $P -I /opt/opendde/kit/opendde/stock/check_pins.py --quiet \
 && $P -m pip freeze --all > /opt/opendde/environment/image-freeze.txt \
 && L=$($P -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") \
 && if [ -n "$PRUNE_GPU_ARCH" ]; then \
      python3 /opt/opendde/container/prune_rocm_arch.py --keep "$PRUNE_GPU_ARCH" \
        "$L/torch/lib/hipblaslt" "$L/torch/lib/rocblas" "$L/torch/lib/aotriton.images" "$L/torch/share/miopen" \
        | tee /opt/opendde/rocm-pruned.txt ; \
    fi \
 && du -sh "$L/torch" "$L/triton" >> /opt/opendde/rocm-pruned.txt

# ------------------------------------------------------------------------------------------------ 5. the port: scripts, patches, tools, tests, entry point
# provenance, shown by `opendde version` (the CI build passes the repository and the commit)
ARG SOURCE_URL=https://github.com/gbouras13/setonix-ports
ARG SOURCE_REV=unknown
COPY scripts/ /opt/opendde/scripts/
COPY patches/ /opt/opendde/patches/
COPY tools/ /opt/opendde/tools/
COPY tests/ /opt/opendde/tests/
COPY container/opendde container/opendde_check.py /opt/opendde/container/
RUN chmod 0755 /opt/opendde/scripts/*.sh \
 && install -m 0755 /opt/opendde/container/opendde /usr/local/bin/opendde \
 && echo "source $SOURCE_URL @ $SOURCE_REV" > /opt/opendde/source.txt \
 && chmod -R a+rX /opt/opendde /opt/python

# /usr/local/bin FIRST: the venv carries upstream's own `opendde` console script (a Click CLI), and with the venv first it answered
# `singularity exec <sif> opendde ...` in place of the port's entry ("No such command 'check'" in the pull job's self-check of
# sha256:89f72c58, 2026-10-17). The entry still puts the venv first for everything it runs.
ENV PATH=/usr/local/bin:/opt/opendde/venv/bin:/opt/rocm/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 ODDE_ROOT=/opt/opendde
# The self-check under the image's own PATH, as `singularity exec <sif> opendde check` runs it.
RUN if [ "$(command -v opendde)" != /usr/local/bin/opendde ]; then \
      echo "PATH: 'opendde' is $(command -v opendde), not the port's entry" >&2; exit 1; \
    fi \
 && opendde check --no-gpu --patch-tests \
 && opendde help > /dev/null \
 && opendde version
LABEL org.opencontainers.image.title="opendde-setonix" \
      org.opencontainers.image.description="OpenDDE 1.1.1 on the optimisation kit's big line (one GCD or row-sharded over up to 8) with the port's eight opt-in fixes, on AMD MI250X (gfx90a), ROCm 7.2.4, torch 2.14.0. No weights: opendde fetch-weights." \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.revision="${SOURCE_REV}" \
      org.opencontainers.image.licenses="Apache-2.0"
WORKDIR /opt/opendde
ENTRYPOINT ["/usr/local/bin/opendde"]
CMD ["help"]
