# syntax=docker/dockerfile:1.7
# ESMFold2 for Pawsey Setonix (AMD Instinct MI250X, gfx90a), with the optimisation kit's modes and the gfx90a fixes that make them run
# (and `exact` byte-identical to stock). The image IS the native port's tree, at /opt/esmfold2:
#   venv/     torch 2.14.0+rocm7.2 + triton-rocm 3.8.0 (download.pytorch.org/whl/rocm7.2) and the rest of the kit's lock, pinned
#             exactly by environment/requirements.lock (the native venv's freeze) and installed --no-deps; Biohub's esm 3.3.0 and
#             transformers fork 4.57.6 from git at the kit's pinned commits; the kit's two packages (opt_core, esmfold2_opt) editable,
#             as the kit's own `run.sh install` does;
#   kit/      the optimisation kit at its pinned commit (esmfold2/, common/);
#   scripts/  the port's environment, routes, run.sh wrapper, weights stage and fetch, boot fixes and nvidia-smi shim, verbatim from
#             the native tree (af3-setonix/esmfold2/); tools/ its output comparison; inputs/ the 1BRS example for `esmfold2 check`.
# NO WEIGHTS: `esmfold2 fetch-weights DIR` downloads the three pinned snapshots (27.5 GB, Hugging Face biohub) and runs the kit's own
# sha256 check on them; `esmfold2 pred` stages them on node-local disk once per job.
#
# The gfx90a fixes (af3-setonix/ESMFOLD2_PORT.md), none of them an edit to a kit or upstream file:
#   - two boot words, applied in memory per run (scripts/boot/sitecustomize.py): `dotprec` (upstream's fused pair-bias kernel asks
#     Triton for tf32x3, which the AMD backend rejects: mapped to ieee) and `smem` (torch on ROCm has no shared_memory_per_block_optin,
#     so the kit assumed an H100's 227 KB: the card's own 64 KB);
#   - the kit's own ablation switch for the levers its 64 KB policy then refuses (MODEL_OPT_LEVERS_OFF; `glue` too on the full model,
#     `tx` too on `fast`: its NaN above 256 tokens);
#   - HIP_LAUNCH_BLOCKING=1 on every route (ROCm 7.2 crashes asynchronously without it), DEBUG_CLR_GRAPH_PACKET_CAPTURE=0, and for
#     exact / fast PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False (the kit's expandable segments never give memory back on ROCm 7.2,
#     so a process folding many inputs ran out); all set by scripts/ef2.sh and scripts/ef2-env.sh.
#   - the kit's run.sh refuses this stack at its image-layer pin check (the CUDA accelerated layer): scripts/ef2-run.sh runs a copy
#     with exactly two lines changed and checks that nothing else differs.
#
# Run on Setonix WITHOUT --rocm (it binds the host's ROCm 6.3.0 libraries, same sonames). `esmfold2 check` fails if any GPU library was
# loaded from outside the image. Images cannot be built on Setonix; GitHub Actions builds this one (.github/workflows/esmfold2-image.yml).
# Context: ports/esmfold2/.

ARG BASE_IMAGE=rocm/dev-ubuntu-24.04:7.2.4@sha256:bdc8e61026cbb844ede93d44d2c50055f51ebb2041906b60182bf3bee3139054
FROM ${BASE_IMAGE}
# Each build argument is declared just before the step that uses it: every RUN after an ARG sees it in its environment, so a changed
# value (SOURCE_REV changes with every commit) re-runs only the steps from its declaration on, and the CI build's layer cache holds.
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# ------------------------------------------------------------------------------------------------ 1. tools
# No ROCm math libraries are installed: the torch 2.14.0+rocm7.2 wheel carries its own complete ROCm 7.2 runtime in torch/lib, and that
# is the one the native port runs on too. git fetches the kit and the lock's three git pins; curl the interpreter and (at run time,
# fetch-weights) the weights; build-essential builds the lock's sdists, as natively; util-linux gives the weights stage its flock.
RUN mkdir -p /opt/esmfold2 \
 && apt-get update --quiet \
 && apt-get install --yes --quiet --no-install-recommends git curl ca-certificates python3 build-essential util-linux \
 && rm -rf /var/lib/apt/lists/* \
 && dpkg-query -W -f '${Package} ${Version}\n' rocm-core hip-runtime-amd hsa-rocr comgr \
      > /opt/esmfold2/rocm-packages.txt

# ------------------------------------------------------------------------------------------------ 2. Python 3.12.12
# The native venv's base is cray-python 3.12.12; this is the same version from python-build-standalone (the boltz2 / colabfold images'
# build); x86_64_v3 covers Setonix's Trento (Zen 3). The validation job proves the bytes do not depend on which build of it this is.
ARG PYTHON_URL=https://github.com/astral-sh/python-build-standalone/releases/download/20260211/cpython-3.12.12%2B20260211-x86_64_v3-unknown-linux-gnu-install_only.tar.gz
ARG PYTHON_SHA256=e10e333796fab1a7793171ea5c2b8cadc6f9c5dfdcc06815fa402c05cbd76d81
RUN curl -fsSL -o /tmp/python.tar.gz "$PYTHON_URL" \
 && echo "$PYTHON_SHA256  /tmp/python.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/python.tar.gz -C /tmp && mv /tmp/python /opt/python && rm -f /tmp/python.tar.gz \
 && /opt/python/bin/python3.12 -V

# ------------------------------------------------------------------------------------------------ 3. the kit, at its pin (sparse)
ARG KIT_REPO=https://github.com/anthropics/uplifting-biomolecular-modeling.git
ARG KIT_COMMIT=f4f62fa6592ae4938d49b1757bea0cfeff9f468e
RUN git clone --quiet --filter=blob:none --no-checkout "$KIT_REPO" /opt/esmfold2/kit \
 && git -C /opt/esmfold2/kit sparse-checkout set --cone esmfold2 common \
 && git -C /opt/esmfold2/kit fetch --quiet --filter=blob:none origin "$KIT_COMMIT" \
 && git -C /opt/esmfold2/kit -c advice.detachedHead=false checkout --quiet "$KIT_COMMIT" \
 && test "$(git -C /opt/esmfold2/kit rev-parse HEAD)" = "$KIT_COMMIT" \
 && test -z "$(git -C /opt/esmfold2/kit status --porcelain --untracked-files=no)" \
 && rm -rf /opt/esmfold2/kit/.git \
 && echo "kit $KIT_REPO @ $KIT_COMMIT" > /opt/esmfold2/kit.txt

# ------------------------------------------------------------------------------------------------ 4. the venv, pinned, pruned to gfx90a
# pip at the lock's version first; then every other line of the lock with --no-deps (torch / triton-rocm from the ROCm index, the
# rest from PyPI, the three git pins from git), as scripts/setup.sh did natively; then the kit's two packages editable, as its
# `run.sh install` does (esmfold2_opt's in-tree build backend writes esmfold2_opt_autoload.pth, the kit's ESMFOLD2_OPT hook). Then
# three checks, any failure fails the build: the image's freeze equals the lock line for line (beside the two -e lines, the kit's
# packages, which pip names by path here: the image's kit/ carries no .git);
# the kit's own pin check, through scripts/efs_pins.py (the forks at their pinned commits; the CUDA accelerated layer reported
# absent, as it is natively); and the upstream switches that layer would flip are off.
#
# The prune: of the ~14 GiB in torch/, ~6 GiB are GPU kernels for cards that are not this one (hipBLASLt, rocBLAS, aotriton, MIOpen's
# databases); it runs in the same layer as the install, so the layer never holds the unpruned tree.
# CC=gcc: python-build-standalone's interpreter records clang as its compiler, which this image does not carry; the lock's one C
# extension (DockQ's operations module, a scoring tool the folds never import) builds with gcc, as it did natively.
ARG PRUNE_GPU_ARCH=gfx90a
ARG TORCH_INDEX=https://download.pytorch.org/whl/rocm7.2
COPY container/prune_rocm_arch.py /opt/esmfold2/container/
COPY environment/requirements.lock /opt/esmfold2/environment/
COPY scripts/efs_pins.py /opt/esmfold2/scripts/
RUN /opt/python/bin/python3.12 -m venv /opt/esmfold2/venv \
 && P=/opt/esmfold2/venv/bin/python \
 && export CC=gcc CXX=g++ \
 && grep -v -E '^(#|$)' /opt/esmfold2/environment/requirements.lock > /tmp/lock.txt \
 && grep -E '^pip==' /tmp/lock.txt > /tmp/pip.txt \
 && grep -v -E '^pip==' /tmp/lock.txt > /tmp/stack.txt \
 && $P -m pip install --quiet --no-deps --no-compile -r /tmp/pip.txt \
 && $P -m pip install --quiet --no-deps --no-compile \
      --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple -r /tmp/stack.txt \
 && $P -m pip install --quiet --no-deps --no-compile -e /opt/esmfold2/kit/common/opt_core -e /opt/esmfold2/kit/esmfold2/opt \
 && $P -m pip freeze --all > /opt/esmfold2/environment/image-freeze.txt \
 && grep -v -E '^(-e |#)' /opt/esmfold2/environment/image-freeze.txt | sort > /tmp/got.txt \
 && sort /tmp/lock.txt > /tmp/want.txt \
 && { diff /tmp/want.txt /tmp/got.txt || { echo "the image's freeze differs from environment/requirements.lock"; exit 1; }; } \
 && test "$(grep -c -E '^-e /opt/esmfold2/kit/(common/opt_core|esmfold2/opt)$' /opt/esmfold2/environment/image-freeze.txt)" = 2 \
 && rm -f /tmp/lock.txt /tmp/pip.txt /tmp/stack.txt /tmp/got.txt /tmp/want.txt \
 && L=$($P -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") \
 && test -f "$L/esmfold2_opt_autoload.pth" \
 && $P -I /opt/esmfold2/scripts/efs_pins.py /opt/esmfold2/kit/esmfold2 \
 && $P -c "from transformers.models.esmfold2 import modeling_esmfold2_common as c; from transformers.models.esmc import modeling_esmc as e; \
assert not c.FLASH_ATTN_AVAILABLE and not e._te_available, 'a CUDA accelerated-layer switch is on'; print('upstream fallbacks: sdpa, torch MLP, torch rotary')" \
 && if [ -n "$PRUNE_GPU_ARCH" ]; then \
      python3 /opt/esmfold2/container/prune_rocm_arch.py --keep "$PRUNE_GPU_ARCH" \
        "$L/torch/lib/hipblaslt" "$L/torch/lib/rocblas" "$L/torch/lib/aotriton.images" "$L/torch/share/miopen" \
        | tee /opt/esmfold2/rocm-pruned.txt ; \
    fi \
 && du -sh "$L/torch" "$L/triton" >> /opt/esmfold2/rocm-pruned.txt

# ------------------------------------------------------------------------------------------------ 5. the port: scripts, tools, entry point
# provenance, shown by `esmfold2 version` (the CI build passes the repository and the commit)
ARG SOURCE_URL=https://github.com/gbouras13/setonix-ports
ARG SOURCE_REV=unknown
COPY scripts/ /opt/esmfold2/scripts/
COPY tools/ /opt/esmfold2/tools/
COPY inputs/ /opt/esmfold2/inputs/
COPY container/esmfold2 container/esmfold2_check.py /opt/esmfold2/container/
RUN chmod 0755 /opt/esmfold2/scripts/bin/nvidia-smi /opt/esmfold2/scripts/*.sh \
 && install -m 0755 /opt/esmfold2/container/esmfold2 /usr/local/bin/esmfold2 \
 && echo "source $SOURCE_URL @ $SOURCE_REV" > /opt/esmfold2/source.txt \
 && chmod -R a+rX /opt/esmfold2 /opt/python

# /usr/local/bin first: `esmfold2` must be the port's entry point whatever the venv's bin holds (the opendde and protenix images once
# reached a venv CLI first). The entry point sets its own PATH inside. The checks run on this final PATH.
ENV PATH=/usr/local/bin:/opt/esmfold2/venv/bin:/opt/rocm/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 EFS_ROOT=/opt/esmfold2
RUN esmfold2 check --no-gpu \
 && esmfold2 help > /dev/null \
 && esmfold2 version
LABEL org.opencontainers.image.title="esmfold2-setonix" \
      org.opencontainers.image.description="ESMFold2 (Biohub esm 3.3.0) with the optimisation kit's modes (exact byte-identical to stock; fast; big on one GCD and row-sharded over up to 8) on AMD MI250X (gfx90a), ROCm 7.2.4, torch 2.14.0. No weights: esmfold2 fetch-weights." \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.revision="${SOURCE_REV}" \
      org.opencontainers.image.licenses="Apache-2.0"
WORKDIR /opt/esmfold2
ENTRYPOINT ["/usr/local/bin/esmfold2"]
CMD ["help"]
