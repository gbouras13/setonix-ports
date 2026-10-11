# syntax=docker/dockerfile:1.7
# Boltz-2 for Pawsey Setonix (AMD Instinct MI250X, gfx90a), with the optimisation kit's modes and the gfx90a recipes that make every one
# of them run (and `exact` byte-identical to stock). The image IS the native port's tree, at /opt/boltz2:
#   venv/     torch 2.14.0+rocm7.2 + triton-rocm 3.8.0 (download.pytorch.org/whl/rocm7.2), numpy 1.26.4 and boltz 2.2.1's stack, pinned
#             exactly by environment/requirements.lock and installed --no-deps; boltz 2.2.1 from the kit's own wheel; the kit's two
#             packages (opt_core, boltz2_opt) editable, as natively;
#   kit/      the optimisation kit at its pinned commit (boltz2/, common/);
#   scripts/  the port's environment, launcher, boot patch, recipes, weights fetch and stage, and nvidia-smi shim, verbatim from the
#             native tree (af3-setonix/boltz2/); tools/ its output comparison.
# NO WEIGHTS: like every image here but proteinmpnn's, it reads them from outside. `boltz2 fetch-weights DIR` downloads the four files
# (6.55 GB, Hugging Face boltz-community, MIT) and checks each against the kit's pinned sha256; `boltz2 pred` stages them per job.
#
# The gfx90a recipes (af3-setonix/BOLTZ2_PORT.md), none of them an edit to a kit file:
#   - the kit's modes run kernels off (no cuEquivariance on ROCm) and drop, BY NAME through the kit's own card-drop table, the levers
#     that cannot run here (NVRTC / cuda.bindings units, H100 tile configs over gfx90a's 64 KB LDS, two torch-pinned units);
#   - HIP_LAUNCH_BLOCKING=1 (stock itself segfaults above ~500 tokens without it) and DEBUG_CLR_GRAPH_PACKET_CAPTURE=0 (ROCm 7.2 corrupts
#     the host heap replaying a captured graph: fast's sampler roll-out), both scripts/bz-env.sh defaults;
#   - one-GCD big runs the kit's flash base with ROWS=1 tiles, --n_gpu P the OpenFold3 tiles and RCCL words (scripts/b2s_launch.py).
#
# Run on Setonix WITHOUT --rocm (it binds the host's ROCm 6.3.0 libraries, same sonames). `boltz2 check` fails if any GPU library was
# loaded from outside the image. Images cannot be built on Setonix; GitHub Actions builds this one (.github/workflows/boltz2-image.yml).
# Context: ports/boltz2/.

ARG BASE_IMAGE=rocm/dev-ubuntu-24.04:7.2.4@sha256:bdc8e61026cbb844ede93d44d2c50055f51ebb2041906b60182bf3bee3139054
FROM ${BASE_IMAGE}
# Each build argument is declared just before the step that uses it: every RUN after an ARG sees it in its environment, so a changed
# value (SOURCE_REV changes with every commit) re-runs only the steps from its declaration on, and the CI build's layer cache holds.
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# ------------------------------------------------------------------------------------------------ 1. tools
# No ROCm math libraries are installed: the torch 2.14.0+rocm7.2 wheel carries its own complete ROCm 7.2 runtime in torch/lib, and that
# is the one the native port runs on too (its rocm/7.2.4 module is loaded for tools, not libraries). git and curl fetch the kit, the
# interpreter and (at run time, fetch-weights) the weights; build-essential builds the four sdists of the lock, as natively; python3
# runs the prune script.
RUN mkdir -p /opt/boltz2 \
 && apt-get update --quiet \
 && apt-get install --yes --quiet --no-install-recommends git curl ca-certificates python3 build-essential \
 && rm -rf /var/lib/apt/lists/* \
 && dpkg-query -W -f '${Package} ${Version}\n' rocm-core hip-runtime-amd hsa-rocr comgr \
      > /opt/boltz2/rocm-packages.txt

# ------------------------------------------------------------------------------------------------ 2. Python 3.12
# The native venv's base is the OpenFold3 conda env's CPython 3.12.14, a /scratch tree with no equivalent in an image. This is the same
# minor version from python-build-standalone (the colabfold image's build); x86_64_v3 covers Setonix's Trento (Zen 3). Nothing in the
# port's numerics depends on the interpreter's patch level -- the validation job proves that by comparing bytes with the native tree.
ARG PYTHON_URL=https://github.com/astral-sh/python-build-standalone/releases/download/20260211/cpython-3.12.12%2B20260211-x86_64_v3-unknown-linux-gnu-install_only.tar.gz
ARG PYTHON_SHA256=e10e333796fab1a7793171ea5c2b8cadc6f9c5dfdcc06815fa402c05cbd76d81
RUN curl -fsSL -o /tmp/python.tar.gz "$PYTHON_URL" \
 && echo "$PYTHON_SHA256  /tmp/python.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/python.tar.gz -C /tmp && mv /tmp/python /opt/python && rm -f /tmp/python.tar.gz \
 && /opt/python/bin/python3.12 -V

# ------------------------------------------------------------------------------------------------ 3. the kit, at its pin (sparse)
ARG KIT_REPO=https://github.com/anthropics/uplifting-biomolecular-modeling.git
ARG KIT_COMMIT=f4f62fa6592ae4938d49b1757bea0cfeff9f468e
RUN git clone --quiet --filter=blob:none --no-checkout "$KIT_REPO" /opt/boltz2/kit \
 && git -C /opt/boltz2/kit sparse-checkout set --cone boltz2 common \
 && git -C /opt/boltz2/kit fetch --quiet --filter=blob:none origin "$KIT_COMMIT" \
 && git -C /opt/boltz2/kit -c advice.detachedHead=false checkout --quiet "$KIT_COMMIT" \
 && test "$(git -C /opt/boltz2/kit rev-parse HEAD)" = "$KIT_COMMIT" \
 && test -z "$(git -C /opt/boltz2/kit status --porcelain --untracked-files=no)" \
 && rm -rf /opt/boltz2/kit/.git \
 && echo "kit $KIT_REPO @ $KIT_COMMIT" > /opt/boltz2/kit.txt

# ------------------------------------------------------------------------------------------------ 4. the venv, pinned, pruned to gfx90a
# setuptools and wheel first, at the native venv's versions; then the lock's wheels (--no-deps), then its four sdists (fairscale,
# antlr4-python3-runtime, ihm, modelcif: no cp312 wheels) against what is installed, with --no-build-isolation, as scripts/setup-venv.sh
# did natively; boltz 2.2.1 from the kit's wheel (its torch / cuEquivariance extras must not be honoured); the kit's two packages
# editable (boltz2_opt's build ships boltz2_opt_autoload.pth, the kit's BOLTZ2_OPT hook, as natively). `pip check` is filtered: boltz
# declares wandb 0.18.7, types-requests, pyyaml 6.0.2 and requests 2.32.3, runs on pyyaml 6.0.3 and requests 2.34.2 (as natively) and
# never imports the other two; those four lines are expected, anything else fails the build.
# The kit's own check_pins then states boltz is installed at its pin, file for file.
#
# The prune: of the ~14 GiB in torch/, ~6 GiB are GPU kernels for cards that are not this one (hipBLASLt, rocBLAS, aotriton, MIOpen's
# databases); it runs in the same layer as the install, so the layer never holds the unpruned tree. The large .so files stay: they are
# DT_NEEDED entries of libtorch_hip.so.
ARG PRUNE_GPU_ARCH=gfx90a
ARG TORCH_INDEX=https://download.pytorch.org/whl/rocm7.2
COPY container/prune_rocm_arch.py /opt/boltz2/container/
COPY environment/requirements.lock /opt/boltz2/environment/
RUN /opt/python/bin/python3.12 -m venv /opt/boltz2/venv \
 && P=/opt/boltz2/venv/bin/python \
 && SDISTS='^(fairscale|antlr4-python3-runtime|ihm|modelcif)==' \
 && grep -v -E "$SDISTS" /opt/boltz2/environment/requirements.lock > /tmp/wheels.txt \
 && grep -E "$SDISTS" /opt/boltz2/environment/requirements.lock > /tmp/sdists.txt \
 && $P -m pip install --quiet --no-deps --no-compile setuptools==81.0.0 wheel==0.48.0 \
 && $P -m pip install --quiet --no-deps --no-compile \
      --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple -r /tmp/wheels.txt \
 && $P -m pip install --quiet --no-deps --no-compile --no-build-isolation -r /tmp/sdists.txt \
 && $P -m pip install --quiet --no-deps --no-compile /opt/boltz2/kit/boltz2/stock/boltz-2.2.1-py3-none-any.whl \
 && $P -m pip install --quiet --no-deps --no-compile --no-build-isolation \
      -e /opt/boltz2/kit/common/opt_core -e /opt/boltz2/kit/boltz2/opt \
 && rm -f /tmp/wheels.txt /tmp/sdists.txt \
 && { $P -m pip check > /tmp/pipcheck.txt 2>&1 || true; } \
 && grep -v -E "^boltz 2\.2\.1 (requires (types-requests|wandb), which is not installed|has requirement (pyyaml==6\.0\.2|requests==2\.32\.3), but you have )" /tmp/pipcheck.txt \
      | grep -v -i -E "^No broken requirements found" > /tmp/pipcheck.unexpected || true \
 && cat /tmp/pipcheck.txt \
 && { [ ! -s /tmp/pipcheck.unexpected ] || { echo "UNEXPECTED pip check lines:"; cat /tmp/pipcheck.unexpected; exit 1; }; } \
 && rm -f /tmp/pipcheck.txt /tmp/pipcheck.unexpected \
 && $P -I /opt/boltz2/kit/boltz2/stock/check_pins.py --quiet \
 && $P -m pip freeze --all > /opt/boltz2/environment/image-freeze.txt \
 && L=$($P -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") \
 && if [ -n "$PRUNE_GPU_ARCH" ]; then \
      python3 /opt/boltz2/container/prune_rocm_arch.py --keep "$PRUNE_GPU_ARCH" \
        "$L/torch/lib/hipblaslt" "$L/torch/lib/rocblas" "$L/torch/lib/aotriton.images" "$L/torch/share/miopen" \
        | tee /opt/boltz2/rocm-pruned.txt ; \
    fi \
 && du -sh "$L/torch" "$L/triton" >> /opt/boltz2/rocm-pruned.txt

# ------------------------------------------------------------------------------------------------ 5. the port: scripts, tools, entry point
# provenance, shown by `boltz2 version` (the CI build passes the repository and the commit)
ARG SOURCE_URL=https://github.com/gbouras13/setonix-ports
ARG SOURCE_REV=unknown
COPY scripts/ /opt/boltz2/scripts/
COPY tools/ /opt/boltz2/tools/
COPY tests/ /opt/boltz2/tests/
COPY container/boltz2 container/boltz2_check.py /opt/boltz2/container/
RUN chmod 0755 /opt/boltz2/scripts/bin/nvidia-smi /opt/boltz2/scripts/*.sh \
 && install -m 0755 /opt/boltz2/container/boltz2 /usr/local/bin/boltz2 \
 && echo "source $SOURCE_URL @ $SOURCE_REV" > /opt/boltz2/source.txt \
 && chmod -R a+rX /opt/boltz2 /opt/python \
 && boltz2 check --no-gpu \
 && boltz2 help > /dev/null \
 && boltz2 version

ENV PATH=/opt/boltz2/scripts/bin:/opt/boltz2/venv/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 B2S_ROOT=/opt/boltz2
LABEL org.opencontainers.image.title="boltz2-setonix" \
      org.opencontainers.image.description="Boltz-2 2.2.1 with the optimisation kit's modes (exact byte-identical to stock; big on one GCD and row-sharded over up to 8) on AMD MI250X (gfx90a), ROCm 7.2.4, torch 2.14.0. No weights: boltz2 fetch-weights." \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.revision="${SOURCE_REV}" \
      org.opencontainers.image.licenses="MIT"
WORKDIR /opt/boltz2
ENTRYPOINT ["/usr/local/bin/boltz2"]
CMD ["help"]
