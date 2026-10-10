# syntax=docker/dockerfile:1.7
# ProteinMPNN for Pawsey Setonix (AMD Instinct MI250X, gfx90a), with the optimisation kit's `exact` mode and the three gfx90a fixes that
# make it byte-identical to stock. The image IS the native port's tree, at /opt/proteinmpnn:
#   venv/         torch 2.14.0+rocm7.2 + triton-rocm 3.8.0 (download.pytorch.org/whl/rocm7.2) and numpy 1.26.4, pinned exactly by
#                 environment/requirements.lock (the native venv's freeze), installed --no-deps;
#   kit/          the optimisation kit at its pinned commit (proteinmpnn/, common/), its two packages installed editable;
#   ProteinMPNN/  upstream dauparas/ProteinMPNN at the kit's pin, WITH both weight sets (they live in that repository; 90 MB), their
#                 sha256 checked against the kit's stock/PINS.json by the kit's own `run.sh install --weights`;
#   scripts/      the port's environment, launcher, boot patch, data-parallel driver and nvidia-smi shim, verbatim from the native tree
#                 (af3-setonix/proteinmpnn/); tools/ its comparison scripts.
# So the image needs no weights, no databases and no network: an input directory of PDBs and an output directory are the whole interface.
#
# Three gfx90a fixes make `--mode exact` byte-identical to stock, none of them an edit to a kit file (PROTEINMPNN_PORT.md):
#   1. scripts/pms_worker_boot.py `rocm_draw` -- the kit's fused draw kernel computes cuRAND's uniform transform; torch on ROCm draws
#      through rocRAND, whose transform differs. Applied by default on a ROCm torch.
#   2. --hybrid_gemm 0 -- the kit's own probe refuses it on this card (max |d| 2.6e-6); the kit is right.
#   3. DEBUG_CLR_GRAPH_PACKET_CAPTURE=0 (scripts/pms-env.sh) -- ROCm 7.2 corrupts the host heap when a captured graph is replayed a few
#      hundred times. See rocm-graph-bug/REPORT_AMD.md.
#
# Run on Setonix WITHOUT --rocm (it binds the host's ROCm 6.3.0 libraries, same sonames). `proteinmpnn check` fails if any GPU library
# was loaded from outside the image. Images cannot be built on Setonix; GitHub Actions builds this one
# (.github/workflows/proteinmpnn-image.yml). Context: ports/proteinmpnn/.

ARG BASE_IMAGE=rocm/dev-ubuntu-24.04:7.2.4@sha256:bdc8e61026cbb844ede93d44d2c50055f51ebb2041906b60182bf3bee3139054
FROM ${BASE_IMAGE}
# Each build argument is declared just before the step that uses it: every RUN after an ARG sees it in its environment, so a changed
# value (SOURCE_REV changes with every commit) re-runs only the steps from its declaration on, and the CI build's layer cache holds.
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# ------------------------------------------------------------------------------------------------ 1. tools only
# No ROCm math libraries are installed: the torch 2.14.0+rocm7.2 wheel carries its own complete ROCm 7.2 runtime in torch/lib, and that
# is the one the native port runs on too (its rocm/7.2.4 module is loaded for rocm-smi and friends, not for libraries). git and curl
# fetch the kit, upstream and the interpreter; python3 runs the prune script below.
RUN mkdir -p /opt/proteinmpnn \
 && apt-get update --quiet \
 && apt-get install --yes --quiet --no-install-recommends git curl ca-certificates python3 \
 && rm -rf /var/lib/apt/lists/* \
 && dpkg-query -W -f '${Package} ${Version}\n' rocm-core hip-runtime-amd hsa-rocr comgr \
      > /opt/proteinmpnn/rocm-packages.txt

# ------------------------------------------------------------------------------------------------ 2. Python 3.11
# The native venv's base is cray-python 3.11.7 (/opt/cray/pe/python), a Cray system path with no equivalent in an image. This is the
# same minor version from python-build-standalone; x86_64_v3 covers Setonix's Trento (Zen 3). Nothing in the port's numerics depends on
# the interpreter's patch level -- the validation job proves that by comparing bytes with the native tree.
ARG PYTHON_URL=https://github.com/astral-sh/python-build-standalone/releases/download/20260211/cpython-3.11.14%2B20260211-x86_64_v3-unknown-linux-gnu-install_only.tar.gz
ARG PYTHON_SHA256=09fe6987211537e503f3f1e5adff582e2aee27846153aac7d5ab91fb309cd634
RUN curl -fsSL -o /tmp/python.tar.gz "$PYTHON_URL" \
 && echo "$PYTHON_SHA256  /tmp/python.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/python.tar.gz -C /tmp && mv /tmp/python /opt/python && rm -f /tmp/python.tar.gz \
 && /opt/python/bin/python3.11 -V

# ------------------------------------------------------------------------------------------------ 3. the kit, at its pin (sparse)
ARG KIT_REPO=https://github.com/anthropics/uplifting-biomolecular-modeling.git
ARG KIT_COMMIT=f4f62fa6592ae4938d49b1757bea0cfeff9f468e
RUN git clone --quiet --filter=blob:none --no-checkout "$KIT_REPO" /opt/proteinmpnn/kit \
 && git -C /opt/proteinmpnn/kit sparse-checkout set --cone proteinmpnn common \
 && git -C /opt/proteinmpnn/kit fetch --quiet --filter=blob:none origin "$KIT_COMMIT" \
 && git -C /opt/proteinmpnn/kit -c advice.detachedHead=false checkout --quiet "$KIT_COMMIT" \
 && test "$(git -C /opt/proteinmpnn/kit rev-parse HEAD)" = "$KIT_COMMIT" \
 && test -z "$(git -C /opt/proteinmpnn/kit status --porcelain --untracked-files=no)" \
 && rm -rf /opt/proteinmpnn/kit/.git \
 && echo "kit $KIT_REPO @ $KIT_COMMIT" > /opt/proteinmpnn/kit.txt

# ------------------------------------------------------------------------------------------------ 4. the venv, pinned, pruned to gfx90a
# --no-deps: every package at the version the native port validated. Then the kit's two packages editable, as natively.
#
# The prune matters here more than in the other images, because the torch wheel ships a whole ROCm user space of its own: of the 14 GiB
# in torch/, 6.2 GiB are GPU kernels for cards that are not this one (hipBLASLt 4.60 -> 0.21 GiB, rocBLAS 0.63 -> 0.31, aotriton 0.86 ->
# 0.10, MIOpen's databases 0.76 -> 0.06). It runs in the same layer as the install, so the layer never holds the unpruned tree.
# What is NOT removed: the large .so files themselves (libmagma 1.4 GiB, libMIOpen 1.1, librocsolver 0.8, librocrand 0.8, librccl 0.5,
# librocsparse 0.5). ProteinMPNN uses none of magma, rocsolver, rocsparse or RCCL -- but all six are DT_NEEDED entries of
# libtorch_hip.so, so removing any of them stops `import torch` outright. They stay.
ARG PRUNE_GPU_ARCH=gfx90a
ARG TORCH_INDEX=https://download.pytorch.org/whl/rocm7.2
COPY container/prune_rocm_arch.py /opt/proteinmpnn/container/
COPY environment/requirements.lock /opt/proteinmpnn/environment/
RUN /opt/python/bin/python3.11 -m venv /opt/proteinmpnn/venv \
 && /opt/proteinmpnn/venv/bin/python -m pip install --quiet --no-deps --no-compile \
      --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple \
      -r /opt/proteinmpnn/environment/requirements.lock \
 && /opt/proteinmpnn/venv/bin/python -m pip install --quiet --no-deps --no-compile \
      -e /opt/proteinmpnn/kit/common/opt_core -e /opt/proteinmpnn/kit/proteinmpnn/opt \
 && /opt/proteinmpnn/venv/bin/python -m pip check \
 && /opt/proteinmpnn/venv/bin/python -m pip freeze --all > /opt/proteinmpnn/environment/image-freeze.txt \
 && L=$(/opt/proteinmpnn/venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") \
 && if [ -n "$PRUNE_GPU_ARCH" ]; then \
      python3 /opt/proteinmpnn/container/prune_rocm_arch.py --keep "$PRUNE_GPU_ARCH" \
        "$L/torch/lib/hipblaslt" "$L/torch/lib/rocblas" "$L/torch/lib/aotriton.images" "$L/torch/share/miopen" \
        | tee /opt/proteinmpnn/rocm-pruned.txt ; \
    fi \
 && du -sh "$L/torch" "$L/triton" >> /opt/proteinmpnn/rocm-pruned.txt

# ------------------------------------------------------------------------------------------------ 5. upstream, at the kit's pin, with its weights
# The clone, then the kit's own install verb: it keeps the checkout and digests both weight sets against stock/PINS.json. check_pins then
# states, for each variant, that the checkout is at the pinned commit and that the weights file a pass loads has the pinned sha256. No
# GPU is needed for either, so a wrong pin fails the build rather than the first job.
ARG MPNN_REPO=https://github.com/dauparas/ProteinMPNN
ARG MPNN_COMMIT=8907e6671bfbfc92303b5f79c4b5e6ce47cdef57
# Unlike the kit's, this clone keeps its .git: check_pins reads the checkout's HEAD to state which commit the weights belong to, and the
# native tree has it too (REBUILD.md). `git gc` packs it down; the weights are the bulk either way.
RUN git clone --quiet "$MPNN_REPO" /opt/proteinmpnn/ProteinMPNN \
 && git -C /opt/proteinmpnn/ProteinMPNN -c advice.detachedHead=false checkout --quiet "$MPNN_COMMIT" \
 && test "$(git -C /opt/proteinmpnn/ProteinMPNN rev-parse HEAD)" = "$MPNN_COMMIT" \
 && git -C /opt/proteinmpnn/ProteinMPNN gc --quiet --aggressive --prune=all \
 && echo "ProteinMPNN $MPNN_REPO @ $MPNN_COMMIT" > /opt/proteinmpnn/mpnn.txt \
 && export MODEL_OPT=/opt/proteinmpnn/kit/proteinmpnn MPNN_DIR=/opt/proteinmpnn/ProteinMPNN \
           PATH=/opt/proteinmpnn/venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 \
 && for v in vanilla soluble; do \
      bash /opt/proteinmpnn/kit/proteinmpnn/run.sh install --weights "$MPNN_DIR" --variant "$v" || exit 1 ; \
      pins=$(/opt/proteinmpnn/venv/bin/python -I /opt/proteinmpnn/kit/proteinmpnn/stock/check_pins.py --variant "$v") || exit 1 ; \
      echo "$pins" | tee -a /opt/proteinmpnn/pins.txt ; \
    done

# ------------------------------------------------------------------------------------------------ 6. the port: scripts, tools, entry point
# provenance, shown by `proteinmpnn version` (the CI build passes the repository and the commit)
ARG SOURCE_URL=https://github.com/gbouras13/setonix-ports
ARG SOURCE_REV=unknown
COPY scripts/ /opt/proteinmpnn/scripts/
COPY tools/ /opt/proteinmpnn/tools/
COPY tests/ /opt/proteinmpnn/tests/
COPY container/proteinmpnn container/proteinmpnn_check.py /opt/proteinmpnn/container/
RUN chmod 0755 /opt/proteinmpnn/scripts/bin/nvidia-smi /opt/proteinmpnn/scripts/*.sh \
 && install -m 0755 /opt/proteinmpnn/container/proteinmpnn /usr/local/bin/proteinmpnn \
 && echo "source $SOURCE_URL @ $SOURCE_REV" > /opt/proteinmpnn/source.txt \
 && chmod -R a+rX /opt/proteinmpnn /opt/python \
 && proteinmpnn check --no-gpu \
 && proteinmpnn help > /dev/null \
 && proteinmpnn version

ENV PATH=/opt/proteinmpnn/scripts/bin:/opt/proteinmpnn/venv/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PMS_ROOT=/opt/proteinmpnn
LABEL org.opencontainers.image.title="proteinmpnn-setonix" \
      org.opencontainers.image.description="ProteinMPNN (dauparas 8907e667) with the optimisation kit's exact mode, byte-identical to stock on AMD MI250X (gfx90a), ROCm 7.2.4, torch 2.14.0. Weights included (MIT, from the upstream repository)." \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.revision="${SOURCE_REV}" \
      org.opencontainers.image.licenses="MIT"
WORKDIR /opt/proteinmpnn
ENTRYPOINT ["/usr/local/bin/proteinmpnn"]
CMD ["help"]
