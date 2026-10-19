# syntax=docker/dockerfile:1.7
# Protenix v2 for Pawsey Setonix (AMD Instinct MI250X, gfx90a), with the optimisation kit's `big` line -- on one GCD, and row-sharded over
# up to 8 -- and the gfx90a recipe that runs it (on one GCD, stock's arithmetic bit for bit). The image IS the native port's tree, at
# /opt/protenix:
#   venv/     torch 2.14.0+rocm7.2 + triton-rocm 3.8.0 (download.pytorch.org/whl/rocm7.2), numpy 2.5.3 and Protenix 2.0.0's inference
#             stack, pinned exactly by environment/requirements.lock and installed --no-deps; protenix 2.0.0 from the kit's own wheel; the
#             kit's two packages (opt_core, protenix_opt) editable, as natively; the port's start-up hooks (hooks/) in site-packages;
#   kit/      the optimisation kit at its pinned commit (protenix_v2/, common/);
#   scripts/  the port's environment, verbatim from the native tree (af3-setonix/protenix/scripts/ptx-env.sh), and its launcher;
#             tools/ the output comparisons.
# NO WEIGHTS: like every image here but proteinmpnn's, it reads them from outside. `protenix fetch-weights DIR` fetches the seven files
# (2.5 GB) as the native port did, each checked against the kit's pinned sha256: the protenix-v2 checkpoint from a Hugging Face mirror
# (upstream's own URL answers HTTP 403; tools/fetch_checkpoint.sh), the six data caches through the kit's own routine from upstream.
#
# The gfx90a recipe (af3-setonix/PROTENIX_PORT.md, "The working configuration"), none of it an edit to a kit or Protenix file:
#   - PROTENIX_OPT_FORCE=1 (the kit identifies the GPU with nvidia-smi alone), LAYERNORM_TYPE=torch (stock JIT-builds a CUDA LayerNorm
#     otherwise), a preset MODEL_OPT_STACK_KEY, and four levers dropped by name through the kit's own MODEL_OPT_LEVERS_OFF:
#     fastln_prebuilt, ln_core and sampler_fuse cannot run here; transition_core builds on gfx90a and returns NaN (PTX inline asm);
#   - HIP_LAUNCH_BLOCKING=1 (rocBLAS's gfx90a gap);
#   - --n_gpu P: the engine's DiT attention, opt_core's fused triangle attention at IEEE precision with the OpenFold3 port's gfx90a tiles,
#     two RCCL settings, and stock's triangle-attention chunk capped at 4 rows; one GCD: expandable segments off (on ROCm 7.2 they strand
#     freed memory, and half the one-GCD ladder failed on it).
#   scripts/ptx-env.sh and scripts/ptx-pred.sh set them, each with its reason and job number.
#
# Run on Setonix WITHOUT --rocm (it binds the host's ROCm 6.3.0 libraries, same sonames). `protenix check` fails if any GPU library was
# loaded from outside the image. Images cannot be built on Setonix; GitHub Actions builds this one (.github/workflows/protenix-image.yml).
# Context: ports/protenix/.

ARG BASE_IMAGE=rocm/dev-ubuntu-24.04:7.2.4@sha256:bdc8e61026cbb844ede93d44d2c50055f51ebb2041906b60182bf3bee3139054
FROM ${BASE_IMAGE}
# Each build argument is declared just before the step that uses it: every RUN after an ARG sees it in its environment, so a changed
# value (SOURCE_REV changes with every commit) re-runs only the steps from its declaration on, and the CI build's layer cache holds.
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# ------------------------------------------------------------------------------------------------ 1. tools
# No ROCm math libraries are installed: the torch 2.14.0+rocm7.2 wheel carries its own complete ROCm 7.2 runtime in torch/lib, and that
# is the one the native port runs on too (its rocm/7.2.4 module is loaded for tools, not libraries). git and curl fetch the kit and the
# interpreter, and the weights at run time (fetch-weights); python3 runs the prune script. build-essential is for Triton, not for the
# install (every package of the lock is a wheel): Triton compiles its kernel launcher with the system C compiler when it first builds a
# kernel -- the fused triangle attention of --n_gpu P, the kit's other Triton levers -- as it does natively with Setonix's gcc.
RUN mkdir -p /opt/protenix \
 && apt-get update --quiet \
 && apt-get install --yes --quiet --no-install-recommends git curl ca-certificates python3 build-essential \
 && rm -rf /var/lib/apt/lists/* \
 && dpkg-query -W -f '${Package} ${Version}\n' rocm-core hip-runtime-amd hsa-rocr comgr \
      > /opt/protenix/rocm-packages.txt

# ------------------------------------------------------------------------------------------------ 2. Python 3.12
# The native venv's base is the OpenFold3 conda env's CPython 3.12.14, a /scratch tree with no equivalent in an image. This is the same
# minor version from python-build-standalone (the colabfold and boltz2 images' build); x86_64_v3 covers Setonix's Trento (Zen 3). Nothing
# in the port's numerics depends on the interpreter's patch level -- the validation job proves that by comparing bytes with the native
# tree.
ARG PYTHON_URL=https://github.com/astral-sh/python-build-standalone/releases/download/20260211/cpython-3.12.12%2B20260211-x86_64_v3-unknown-linux-gnu-install_only.tar.gz
ARG PYTHON_SHA256=e10e333796fab1a7793171ea5c2b8cadc6f9c5dfdcc06815fa402c05cbd76d81
RUN curl -fsSL -o /tmp/python.tar.gz "$PYTHON_URL" \
 && echo "$PYTHON_SHA256  /tmp/python.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/python.tar.gz -C /tmp && mv /tmp/python /opt/python && rm -f /tmp/python.tar.gz \
 && /opt/python/bin/python3.12 -V

# ------------------------------------------------------------------------------------------------ 3. the kit, at its pin (sparse)
ARG KIT_REPO=https://github.com/anthropics/uplifting-biomolecular-modeling.git
ARG KIT_COMMIT=f4f62fa6592ae4938d49b1757bea0cfeff9f468e
RUN git clone --quiet --filter=blob:none --no-checkout "$KIT_REPO" /opt/protenix/kit \
 && git -C /opt/protenix/kit sparse-checkout set --cone protenix_v2 common \
 && git -C /opt/protenix/kit fetch --quiet --filter=blob:none origin "$KIT_COMMIT" \
 && git -C /opt/protenix/kit -c advice.detachedHead=false checkout --quiet "$KIT_COMMIT" \
 && test "$(git -C /opt/protenix/kit rev-parse HEAD)" = "$KIT_COMMIT" \
 && test -z "$(git -C /opt/protenix/kit status --porcelain --untracked-files=no)" \
 && rm -rf /opt/protenix/kit/.git \
 && echo "kit $KIT_REPO @ $KIT_COMMIT" > /opt/protenix/kit.txt

# ------------------------------------------------------------------------------------------------ 4. the venv, pinned, pruned to gfx90a
# setuptools and wheel first, at the native venv's versions; then the lock's wheels (--no-deps); protenix 2.0.0 from the kit's wheel (its
# torch 2.7.1 / cuEquivariance / deepspeed requirements must not be honoured); the kit's two packages editable (protenix_opt's build ships
# protenix_opt_autoload.pth, the kit's PROTENIX_OPT hook, as natively), all as scripts/setup-venv.sh did natively. The lock is the native
# process's own versions, not protenix's pins: the native port ran, and was validated, on the OpenFold3 env's numpy 2.5.3, rdkit
# 2025.9.6, gemmi 0.7.5, protobuf 7.36.2 and the rest. So `pip check` is filtered: the 22 lines below, all protenix's own requirements --
# the packages it declares but never imports at inference, and the pins the native process runs past -- are expected; anything else
# fails the build. The kit's own check_pins then states protenix is installed at its pin, file for file.
#
# The prune: of the ~14 GiB in torch/, ~6 GiB are GPU kernels for cards that are not this one (hipBLASLt, rocBLAS, aotriton, MIOpen's
# databases); it runs in the same layer as the install, so the layer never holds the unpruned tree. The large .so files stay: they are
# DT_NEEDED entries of libtorch_hip.so.
ARG PRUNE_GPU_ARCH=gfx90a
ARG TORCH_INDEX=https://download.pytorch.org/whl/rocm7.2
COPY container/prune_rocm_arch.py /opt/protenix/container/
COPY environment/requirements.lock /opt/protenix/environment/
RUN /opt/python/bin/python3.12 -m venv /opt/protenix/venv \
 && P=/opt/protenix/venv/bin/python \
 && $P -m pip install --quiet --no-deps --no-compile setuptools==84.0.0 wheel==0.48.0 \
 && $P -m pip install --quiet --no-deps --no-compile \
      --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple -r /opt/protenix/environment/requirements.lock \
 && $P -m pip install --quiet --no-deps --no-compile /opt/protenix/kit/protenix_v2/stock/protenix-2.0.0-py3-none-any.whl \
 && $P -m pip install --quiet --no-deps --no-compile --no-build-isolation \
      -e /opt/protenix/kit/common/opt_core -e /opt/protenix/kit/protenix_v2/opt \
 && { $P -m pip check > /tmp/pipcheck.txt 2>&1 || true; } \
 && NOTINST='torchvision|torchaudio|cuequivariance-ops-torch-cu12|cuequivariance-torch|matplotlib|ipywidgets|py3dmol|modelcif|scikit-learn-extra|deepspeed|triton|icecream|ipdb|wandb' \
 && PINNED='torch==2\.7\.1|tqdm==4\.67\.1|pyyaml==6\.0\.2|rdkit==2025\.9\.3|gemmi==0\.6\.7|pdbeccdutils==1\.0\.0|protobuf==6\.31\.1|numpy==2\.4\.1' \
 && grep -v -i -E "^protenix 2\.0\.0 (requires ($NOTINST), which is not installed|has requirement ($PINNED), but you have )" /tmp/pipcheck.txt \
      | grep -v -i -E "^No broken requirements found" > /tmp/pipcheck.unexpected || true \
 && cat /tmp/pipcheck.txt \
 && { [ ! -s /tmp/pipcheck.unexpected ] || { echo "UNEXPECTED pip check lines:"; cat /tmp/pipcheck.unexpected; exit 1; }; } \
 && rm -f /tmp/pipcheck.txt /tmp/pipcheck.unexpected \
 && $P -I /opt/protenix/kit/protenix_v2/stock/check_pins.py --quiet \
 && $P -m pip freeze --all > /opt/protenix/environment/image-freeze.txt \
 && L=$($P -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") \
 && if [ -n "$PRUNE_GPU_ARCH" ]; then \
      python3 /opt/protenix/container/prune_rocm_arch.py --keep "$PRUNE_GPU_ARCH" \
        "$L/torch/lib/hipblaslt" "$L/torch/lib/rocblas" "$L/torch/lib/aotriton.images" "$L/torch/share/miopen" \
        | tee /opt/protenix/rocm-pruned.txt ; \
    fi \
 && du -sh "$L/torch" "$L/triton" >> /opt/protenix/rocm-pruned.txt

# ------------------------------------------------------------------------------------------------ 5. the port: hooks, scripts, tools, entry point
# The hooks go into site-packages, as scripts/setup-venv.sh puts them there natively: setonix_rankcache.pth gives every rank of the
# sharded line its own Triton / MIOpen / Inductor cache, and arms the env-gated hooks beside it (the triangle-attention chunk cap, the
# deterministic-run and prediction-dump hook, two diagnostics). Not the native 00-pxs-base-site.pth: this venv is not layered.
# provenance, shown by `protenix version` (the CI build passes the repository and the commit)
ARG SOURCE_URL=https://github.com/gbouras13/setonix-ports
ARG SOURCE_REV=unknown
COPY hooks/ /opt/protenix/hooks/
COPY scripts/ /opt/protenix/scripts/
COPY tools/ /opt/protenix/tools/
COPY tests/ /opt/protenix/tests/
COPY container/protenix container/protenix_check.py /opt/protenix/container/
RUN L=$(/opt/protenix/venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") \
 && install -m 0644 /opt/protenix/hooks/*.py /opt/protenix/hooks/setonix_rankcache.pth "$L/" \
 && chmod 0755 /opt/protenix/scripts/*.sh \
 && install -m 0755 /opt/protenix/container/protenix /usr/local/bin/protenix \
 && echo "source $SOURCE_URL @ $SOURCE_REV" > /opt/protenix/source.txt \
 && chmod -R a+rX /opt/protenix /opt/python \
 && protenix check --no-gpu \
 && protenix help > /dev/null \
 && protenix version

# /usr/local/bin first. Stock Protenix's wheel installs its own CLI as venv/bin/protenix, which would otherwise shadow this entry point
# for `singularity exec <sif> protenix ...`: the first image's pull job ran stock's CLI in its self-check, which took stock's CUDA
# LayerNorm JIT and died for want of ninja (job 51048476). The step above ran before this PATH existed, so it could not see that. The
# entry point puts the venv first again inside, where the kit expects stock's CLI. The check below runs on the final PATH.
ENV PATH=/usr/local/bin:/opt/protenix/venv/bin:/opt/rocm/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PXS_ROOT=/opt/protenix
RUN test "$(command -v protenix)" = /usr/local/bin/protenix \
 && test "$(command -v python)" = /opt/protenix/venv/bin/python \
 && protenix version > /dev/null
LABEL org.opencontainers.image.title="protenix-setonix" \
      org.opencontainers.image.description="Protenix v2 (2.0.0) with the optimisation kit's big line, on one GCD (stock's arithmetic bit for bit) and row-sharded over up to 8, on AMD MI250X (gfx90a), ROCm 7.2.4, torch 2.14.0. No weights: protenix fetch-weights." \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.revision="${SOURCE_REV}" \
      org.opencontainers.image.licenses="Apache-2.0"
WORKDIR /opt/protenix
ENTRYPOINT ["/usr/local/bin/protenix"]
CMD ["help"]
