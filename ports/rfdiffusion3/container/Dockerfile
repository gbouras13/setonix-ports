# syntax=docker/dockerfile:1.7
# RFdiffusion3 for Pawsey Setonix (AMD Instinct MI250X, gfx90a), with the optimisation kit's `exact` mode and the two gfx90a fixes that
# make it produce stock's designs. The image IS the native port's tree, at /opt/rfdiffusion3:
#   venv-stock/   the PRISTINE interpreter: `--mode off` runs there, under `python -I`
#   venv-kit/     the same environment with the kit's ten-file rfd3/model overlay applied: `exact` and `fast` run there
#   kit/          the optimisation kit at its pinned commit (rfdiffusion3/, common/), its two packages installed editable in both
#   foundry-4010e3e2/  upstream RFdiffusion3, unpacked from the kit's OWN source archive (stock/foundry-4010e3e.tar.gz) and installed
#                 from it, which is one of the install sources the kit's pin check accepts
#   scripts/, tools/   the port's environment, boot fix, nvidia-smi shim and comparison tools, from the native tree
# This kit is a FILE overlay on the installed `rfd3` package, so the two interpreters must never share a site-packages. They are built
# in one layer and then hardlinked together, so the image carries one copy of every file they have in common.
#
# NO WEIGHTS: the checkpoint is 2.7 GB and upstream distributes it itself, so the image mounts it exactly as the kit's own Dockerfile
# does -- `-B /path/to/weights:/weights`, RFD3_CKPT=/weights/rfd3/rfd3_latest.ckpt. `rfdiffusion3 fetch /weights/rfd3` downloads and
# digest-checks it with the kit's own verb, on a node that has network.
#
# Two gfx90a fixes, neither of them an edit to a kit file (RFDIFFUSION_PORT.md):
#   1. apex, built here from the kit's pinned commit with patches/apex-gfx90a-warp32.patch. Upstream binds apex's FusedRMSNorm as EVERY
#      RMSNorm of the network, and its layer-norm kernel assumes `blockDim.x == warpSize` (32 on NVIDIA, 64 here), so four rows per block
#      read each other's partial sums: 29 % relative error in fp32 and NOT ONE of 16 designs protein-like. With the width passed
#      explicitly it is torch.nn.RMSNorm's own error (1.2-2.3e-7) and 152 of 152 designs are protein-like. A second patch,
#      patches/apex-rocm-build.patch, only makes the pinned sources compile here; it changes no arithmetic.
#   2. RFD_BOOT=fzt_rows_off (scripts/boot/sitecustomize.py, applied in memory at interpreter start). The kit's fused SwiGLU transition
#      fuses every row count at compute capability 9.0 -- what gfx90a reports -- because cuBLAS keeps one summation order there. This
#      card's GEMMs are hipBLASLt's, and with the lever on, `exact` is not stock's designs (one 20.2 A away, a changed residue). The fix
#      gives capability 9.0 an empty served-row range, so the lever takes the stock body BY NAME, which is a route the kit defines for
#      itself; it costs no measurable time. With it, `exact` is byte-identical to stock and 1.37x faster.
# And one setting, DEBUG_CLR_GRAPH_PACKET_CAPTURE=0 (scripts/rfd-env.sh): ROCm 7.2 corrupts the host heap replaying a captured graph.
#
# Run on Setonix WITHOUT --rocm (it binds the host's ROCm 6.3.0 libraries, same sonames). `rfdiffusion3 check` fails if a GPU library
# was loaded from outside the image. Images cannot be built on Setonix; GitHub Actions builds this one
# (.github/workflows/rfdiffusion3-image.yml). Context: ports/rfdiffusion3/.

ARG BASE_IMAGE=rocm/dev-ubuntu-24.04:7.2.4@sha256:bdc8e61026cbb844ede93d44d2c50055f51ebb2041906b60182bf3bee3139054
FROM ${BASE_IMAGE}
# Each build argument is declared just before the step that uses it, so a changed value re-runs only the steps from there on.
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# ------------------------------------------------------------------------------------------------ 1. tools only
# No ROCm math libraries are installed: the torch 2.14.0+rocm7.2 wheel carries its own complete ROCm 7.2 runtime in torch/lib, which is
# what the native port runs on too. The base image's hipcc and clang build apex, and only apex -- g++ is its host compiler (torch 2.14
# compiles extensions as C++20, which Ubuntu 24.04's gcc-13 has). jdupes hardlinks the two interpreters together.
RUN mkdir -p /opt/rfdiffusion3 \
 && apt-get update --quiet \
 && apt-get install --yes --quiet --no-install-recommends git curl ca-certificates python3 g++ jdupes \
 && rm -rf /var/lib/apt/lists/* \
 && dpkg-query -W -f '${Package} ${Version}\n' rocm-core hip-runtime-amd hsa-rocr comgr \
      > /opt/rfdiffusion3/rocm-packages.txt

# ------------------------------------------------------------------------------------------------ 2. Python 3.12
# Upstream requires Python >= 3.12 (`rc-foundry` requires-python), so this port cannot use the 3.11 the others do. The native base is
# cray-python 3.12.12, a Cray system path with no equivalent in an image; this is the same minor version from python-build-standalone,
# x86_64_v3 for Setonix's Trento (Zen 3). The validation job proves the patch level does not matter by comparing bytes with the native tree.
ARG PYTHON_URL=https://github.com/astral-sh/python-build-standalone/releases/download/20260211/cpython-3.12.12%2B20260211-x86_64_v3-unknown-linux-gnu-install_only.tar.gz
ARG PYTHON_SHA256=e10e333796fab1a7793171ea5c2b8cadc6f9c5dfdcc06815fa402c05cbd76d81
RUN curl -fsSL -o /tmp/python.tar.gz "$PYTHON_URL" \
 && echo "$PYTHON_SHA256  /tmp/python.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/python.tar.gz -C /tmp && mv /tmp/python /opt/python && rm -f /tmp/python.tar.gz \
 && /opt/python/bin/python3.12 -V

# ------------------------------------------------------------------------------------------------ 3. the kit, at its pin (sparse)
ARG KIT_REPO=https://github.com/anthropics/uplifting-biomolecular-modeling.git
ARG KIT_COMMIT=f4f62fa6592ae4938d49b1757bea0cfeff9f468e
RUN git clone --quiet --filter=blob:none --no-checkout "$KIT_REPO" /opt/rfdiffusion3/kit \
 && git -C /opt/rfdiffusion3/kit sparse-checkout set --cone rfdiffusion3 common \
 && git -C /opt/rfdiffusion3/kit fetch --quiet --filter=blob:none origin "$KIT_COMMIT" \
 && git -C /opt/rfdiffusion3/kit -c advice.detachedHead=false checkout --quiet "$KIT_COMMIT" \
 && test "$(git -C /opt/rfdiffusion3/kit rev-parse HEAD)" = "$KIT_COMMIT" \
 && test -z "$(git -C /opt/rfdiffusion3/kit status --porcelain --untracked-files=no)" \
 && rm -rf /opt/rfdiffusion3/kit/.git \
 && echo "kit $KIT_REPO @ $KIT_COMMIT" > /opt/rfdiffusion3/kit.txt \
 && tar -xzf /opt/rfdiffusion3/kit/rfdiffusion3/stock/foundry-4010e3e.tar.gz -C /opt/rfdiffusion3 \
 && echo "upstream rc-foundry from the kit's own archive stock/foundry-4010e3e.tar.gz" > /opt/rfdiffusion3/upstream.txt

# ------------------------------------------------------------------------------------------------ 4. apex 0.1, for THIS card
# No public apex wheel exists for this stack, and a kit mode refuses to run without apex (upstream's import guard binds its FusedRMSNorm
# as every RMSNorm; absent, the KERNELS census reads `rmsnorm=fallback` and the pass is refused, exit 5). Only the one extension upstream
# actually uses is built -- apex's own setup.py has no ROCm path at this commit and would build twelve, none of the other eleven imported
# by rfd3. The wheel is built once and installed into both interpreters in step 5.
# The throwaway build venv takes torch --no-deps, so torch's own import-time dependencies have to be named: `setup.py` imports
# torch.utils.cpp_extension, which runs torch/__init__.py, which imports typing_extensions, filelock, sympy, networkx, jinja2 and
# fsspec. They come from the lock, at the versions the native port validated, so this venv and step 5's two agree on every one.
# CC/CXX are set for the same reason the native port's setup.sh sets them: torch takes the host compiler from sysconfig, and this
# interpreter was built with clang, so without them the host .cpp goes to a clang++ that is not installed here.
ARG APEX_REPO=https://github.com/NVIDIA/apex.git
ARG APEX_COMMIT=a1d527a857e8da64c4e7237ca89ec699fb4d9eaf
ARG PYTORCH_ROCM_ARCH=gfx90a
ARG TORCH_INDEX_FOR_APEX=https://download.pytorch.org/whl/rocm7.2
COPY patches/ /opt/rfdiffusion3/patches/
COPY container/apex_fusedln_setup.py /opt/rfdiffusion3/container/
COPY environment/requirements.lock /opt/rfdiffusion3/environment/
RUN git clone --quiet "$APEX_REPO" /tmp/apex \
 && git -C /tmp/apex -c advice.detachedHead=false checkout --quiet "$APEX_COMMIT" \
 && test "$(git -C /tmp/apex rev-parse HEAD)" = "$APEX_COMMIT" \
 && git -C /tmp/apex apply /opt/rfdiffusion3/patches/apex-rocm-build.patch \
 && git -C /tmp/apex apply /opt/rfdiffusion3/patches/apex-gfx90a-warp32.patch \
 && echo "apex $APEX_REPO @ $APEX_COMMIT + patches/apex-rocm-build.patch + patches/apex-gfx90a-warp32.patch" \
      > /opt/rfdiffusion3/apex.txt \
 && /opt/python/bin/python3.12 -m venv /tmp/apexbuild \
 && /tmp/apexbuild/bin/python -m pip install --quiet --no-compile --upgrade "setuptools>=64" wheel ninja==1.13.0 \
 && /tmp/apexbuild/bin/python -m pip install --quiet --no-compile --no-deps \
      --index-url "$TORCH_INDEX_FOR_APEX" --extra-index-url https://pypi.org/simple \
      torch==2.14.0+rocm7.2 \
      $(grep -iE '^(typing_extensions|filelock|sympy|mpmath|networkx|jinja2|markupsafe|fsspec)==' \
          /opt/rfdiffusion3/environment/requirements.lock) \
 && ( cd /tmp/apex && APEX_CPP_EXT=1 APEX_CUDA_EXT=1 CC=gcc CXX=g++ \
        /tmp/apexbuild/bin/python /opt/rfdiffusion3/container/apex_fusedln_setup.py bdist_wheel -d /tmp/apexwheel ) \
 && ls -la /tmp/apexwheel \
 && rm -rf /tmp/apexbuild

# ------------------------------------------------------------------------------------------------ 5. the two interpreters
# Both get the same packages (--no-deps: every package at the version the native port validated), upstream from the kit's own archive,
# apex, and the kit's two packages editable -- the kit's core gate reads [tool.opt_core] from above the package, so a wheel install is
# refused. Then `install apply` puts the ten-file overlay into venv-kit ONLY, and `check_pins` states each tree's state: pristine for
# venv-stock, not pristine for venv-kit (the overlay is in: expected).
#
# The prune: of the torch wheel's ~14 GiB, most is GPU kernels for cards that are not this one. It runs in the same layer as the install,
# so the layer never holds the unpruned tree. What is NOT removed: the large .so files themselves -- rfd3 uses none of magma, rocsolver,
# rocsparse or RCCL, but all are DT_NEEDED entries of libtorch_hip.so and removing any stops `import torch`.
# jdupes then hardlinks the two interpreters: they differ only in the ten overlaid files and their backups, so one copy of the rest.
ARG PRUNE_GPU_ARCH=gfx90a
ARG TORCH_INDEX=https://download.pytorch.org/whl/rocm7.2
ARG FOUNDRY_VERSION=0.2.1.dev13+g4010e3e2e
COPY container/prune_rocm_arch.py container/build_venvs.sh /opt/rfdiffusion3/container/
RUN W=/opt/rfdiffusion3 TORCH_INDEX="$TORCH_INDEX" FOUNDRY_VERSION="$FOUNDRY_VERSION" PRUNE_GPU_ARCH="$PRUNE_GPU_ARCH" \
    APEX_WHEEL_DIR=/tmp/apexwheel sh /opt/rfdiffusion3/container/build_venvs.sh \
 && rm -rf /tmp/apex /tmp/apexwheel

# ------------------------------------------------------------------------------------------------ 6. the port: scripts, tools, entry point
ARG SOURCE_URL=https://github.com/gbouras13/setonix-ports
ARG SOURCE_REV=unknown
COPY scripts/ /opt/rfdiffusion3/scripts/
COPY tools/ /opt/rfdiffusion3/tools/
COPY tests/ /opt/rfdiffusion3/tests/
COPY container/rfdiffusion3 container/rfdiffusion3_check.py /opt/rfdiffusion3/container/
RUN chmod 0755 /opt/rfdiffusion3/scripts/bin/nvidia-smi /opt/rfdiffusion3/scripts/*.sh \
 && install -m 0755 /opt/rfdiffusion3/container/rfdiffusion3 /usr/local/bin/rfdiffusion3 \
 && echo "source $SOURCE_URL @ $SOURCE_REV" > /opt/rfdiffusion3/source.txt \
 && chmod -R a+rX /opt/rfdiffusion3 /opt/python \
 && rfdiffusion3 check --no-gpu \
 && rfdiffusion3 help > /dev/null \
 && rfdiffusion3 version

ENV PATH=/opt/rfdiffusion3/scripts/bin:/opt/rfdiffusion3/venv-kit/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 RFD_ROOT=/opt/rfdiffusion3
LABEL org.opencontainers.image.title="rfdiffusion3-setonix" \
      org.opencontainers.image.description="RFdiffusion3 (RosettaCommons foundry 4010e3e2) with the optimisation kit's exact mode, producing stock's designs on AMD MI250X (gfx90a), ROCm 7.2.4, torch 2.14.0. Checkpoint not included: mount it at /weights." \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.revision="${SOURCE_REV}" \
      org.opencontainers.image.licenses="BSD-3-Clause"
WORKDIR /opt/rfdiffusion3
ENTRYPOINT ["/usr/local/bin/rfdiffusion3"]
CMD ["help"]
