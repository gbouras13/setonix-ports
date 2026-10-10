# syntax=docker/dockerfile:1.7
# ColabFold 1.6.3 (AlphaFold2-Multimer v3) for Pawsey Setonix (AMD Instinct MI250X, gfx90a), with the ColabFold kit's modes and
# colabfold_search on MMseqs2's GPU build. The image IS the native port's tree, at /opt/colabfold:
#   venv/    ColabFold 1.6.3 + alphafold-colabfold 2.3.20 on jax / jaxlib / jax-rocm7-plugin / -pjrt 0.10.2, Python 3.12.12, pinned exactly
#            by environment/requirements.lock (the native venv's freeze);
#   kit/     the optimisation kit at its pinned commit (colabfold/, common/), its two packages installed editable;
#   kitpin/  the kit's stock pin restated for 1.6.3 (tools/repin163.py through scripts/make-kitpin.sh, as natively);
#   scripts/ the port's environment, runner, boot patch and nvidia-smi shim (af3-setonix/COLABFOLD_PORT.md), and cf-search.sh;
#   bin/     MMseqs2's GPU build, copied from the mmseqs2-hip image (MMSEQS_IMAGE, pinned by digest).
# NOT inside: the AlphaFold2 parameters (default: Pawsey's copy under /scratch/references), databases, inputs, outputs.
#
# Run on Setonix WITHOUT --rocm (it binds the host's ROCm 6.3.0 libraries, same sonames). `colabfold check` fails if any GPU library
# was loaded from outside the image. Images cannot be built on Setonix; GitHub Actions builds this one (.github/workflows/colabfold-image.yml).
# Context: ports/colabfold/.

# MMseqs2's GPU build: the mmseqs2-hip image of commit 49d7cbd (run 37941215410), pinned by digest
ARG MMSEQS_IMAGE=ghcr.io/gbouras13/setonix-ports/mmseqs2-hip@sha256:4a92d596be3684ad781dcda8f4c608b4ad5271de647483d72cd696ae27babfd6
ARG BASE_IMAGE=rocm/dev-ubuntu-24.04:7.2.4@sha256:bdc8e61026cbb844ede93d44d2c50055f51ebb2041906b60182bf3bee3139054
FROM ${MMSEQS_IMAGE} AS mmseqs

FROM ${BASE_IMAGE}
# Each build argument is declared just before the step that uses it: every RUN after an ARG sees it in its environment, so a changed
# value (SOURCE_REV changes with every commit) re-runs only the steps from its declaration on, and the CI build's layer cache holds.
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# ------------------------------------------------------------------------------------------------ 1a. ROCm's big GPU math libraries, pruned
# The same ROCm layers as the af3jax image (ports/af3jax/container/Dockerfile): hipBLASLt, rocBLAS and MIOpen pruned to PRUNE_GPU_ARCH in
# the layer that installs them; the rest in the next layer (each well inside ghcr.io's 10 GB per layer and 10 minutes per upload).
ARG ROCM_VERSION=7.2.4
ARG PRUNE_GPU_ARCH=gfx90a
COPY container/prune_rocm_arch.py /opt/colabfold/container/
RUN apt-get update --quiet \
 && apt-get install --yes --quiet --no-install-recommends hipblaslt rocblas hipblas miopen-hip python3 \
 && rm -rf /var/lib/apt/lists/* \
 && v=$(dpkg-query -W -f '${Version}' rocm-core) \
 && case "$v" in "${ROCM_VERSION}".*) echo "ROCm user space: rocm-core $v";; *) echo "rocm-core is $v, not ${ROCM_VERSION}" >&2; exit 1;; esac \
 && if [ -n "$PRUNE_GPU_ARCH" ]; then \
      python3 /opt/colabfold/container/prune_rocm_arch.py --keep "$PRUNE_GPU_ARCH" \
        /opt/rocm/lib/hipblaslt/library /opt/rocm/lib/rocblas/library /opt/rocm/share/miopen/db /opt/rocm/lib/miopen \
      && echo "pruned to $PRUNE_GPU_ARCH: hipBLASLt, rocBLAS, MIOpen kernel files" > /opt/colabfold/rocm-pruned.txt ; \
    fi

# ------------------------------------------------------------------------------------------------ 1b. the other ROCm libraries
# What XLA's ROCm plugin needs beyond the base image's HIP runtime, and libdw (elfutils) the plugin links; git for the kit.
RUN apt-get update --quiet \
 && apt-get install --yes --quiet --no-install-recommends \
      rccl hipfft rocfft hipsparse rocsparse hipsolver rocsolver hiprand rocrand roctracer \
      rocm-llvm rocm-device-libs rocm-smi-lib rocprofiler-sdk libdw1t64 \
      git wget curl ca-certificates zstd \
 && rm -rf /var/lib/apt/lists/* \
 && dpkg-query -W -f '${Package} ${Version}\n' rocm-core hip-runtime-amd hsa-rocr comgr hipblaslt rocblas miopen-hip rccl rocfft rocsolver rocsparse \
      > /opt/colabfold/rocm-packages.txt \
 && { cat /opt/colabfold/rocm-pruned.txt >> /opt/colabfold/rocm-packages.txt 2>/dev/null || true; }

# ------------------------------------------------------------------------------------------------ 2. Python 3.12.12 (the native venv's version)
ARG PYTHON_URL=https://github.com/astral-sh/python-build-standalone/releases/download/20260211/cpython-3.12.12%2B20260211-x86_64_v3-unknown-linux-gnu-install_only.tar.gz
ARG PYTHON_SHA256=e10e333796fab1a7793171ea5c2b8cadc6f9c5dfdcc06815fa402c05cbd76d81
RUN curl -fsSL -o /tmp/python.tar.gz "$PYTHON_URL" \
 && echo "$PYTHON_SHA256  /tmp/python.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/python.tar.gz -C /tmp && mv /tmp/python /opt/python && rm -f /tmp/python.tar.gz \
 && /opt/python/bin/python3.12 -V

# ------------------------------------------------------------------------------------------------ 3. the kit, at its pin (sparse)
ARG KIT_REPO=https://github.com/anthropics/uplifting-biomolecular-modeling.git
ARG KIT_COMMIT=f4f62fa6592ae4938d49b1757bea0cfeff9f468e
RUN git clone --quiet --filter=blob:none --no-checkout "$KIT_REPO" /opt/colabfold/kit \
 && git -C /opt/colabfold/kit sparse-checkout set --cone colabfold common \
 && git -C /opt/colabfold/kit fetch --quiet --filter=blob:none origin "$KIT_COMMIT" \
 && git -C /opt/colabfold/kit -c advice.detachedHead=false checkout --quiet "$KIT_COMMIT" \
 && test "$(git -C /opt/colabfold/kit rev-parse HEAD)" = "$KIT_COMMIT" \
 && rm -rf /opt/colabfold/kit/.git \
 && echo "kit $KIT_REPO @ $KIT_COMMIT" > /opt/colabfold/kit.txt

# ------------------------------------------------------------------------------------------------ 4. the venv, pinned exactly
# The native venv's freeze, with --no-deps: every package at the version the port validated (scripts/setup-venv.sh built it from
# colabfold[alphafold-minus-jax]==1.6.3 plus the jax 0.10.2 ROCm wheels). Then the kit's two packages, editable, as natively.
COPY environment/requirements.lock /opt/colabfold/environment/
RUN /opt/python/bin/python3.12 -m venv /opt/colabfold/venv \
 && /opt/colabfold/venv/bin/python -m pip install --quiet --no-deps -r /opt/colabfold/environment/requirements.lock \
 && /opt/colabfold/venv/bin/python -m pip install --quiet --no-deps -e /opt/colabfold/kit/common/opt_core -e /opt/colabfold/kit/colabfold/opt \
 && /opt/colabfold/venv/bin/python -m pip check \
 && /opt/colabfold/venv/bin/python -m pip freeze --all > /opt/colabfold/environment/image-freeze.txt

# ------------------------------------------------------------------------------------------------ 5. the re-pinned tree (kitpin/)
# scripts/make-kitpin.sh, as natively: the kit's stock pin restated for ColabFold 1.6.3 (tools/repin163.py) from the two PyPI wheels, and
# configs/mi250x.env. Its compile-cache default is a directory under TMPDIR (the image is read-only); `colabfold` sets a per-user one.
COPY tools/ /opt/colabfold/tools/
COPY scripts/ /opt/colabfold/scripts/
RUN mkdir -p /opt/colabfold/wheels \
 && /opt/colabfold/venv/bin/python -m pip download --quiet --no-deps colabfold==1.6.3 alphafold-colabfold==2.3.20 -d /opt/colabfold/wheels \
 && CFS_W=/opt/colabfold CFS_JIT_DEFAULT='${TMPDIR:-/tmp}/colabfold-jit' \
      CFS_REPIN_IMAGE="the colabfold image: python-build-standalone 3.12.12, jax 0.10.2 + jax-rocm7-plugin/pjrt 0.10.2 on AMD's ROCm 7.2.4 packages; ColabFold 1.6.3 + alphafold-colabfold 2.3.20 from PyPI." \
      bash /opt/colabfold/scripts/make-kitpin.sh \
 && rm -rf /opt/colabfold/wheels \
 && cd /opt/colabfold/kitpin && /opt/colabfold/venv/bin/python -I stock/check_pins.py --checks packages,files

# ------------------------------------------------------------------------------------------------ 6. MMseqs2's GPU build
COPY --from=mmseqs /opt/mmseqs/bin/mmseqs /opt/mmseqs/bin/mmseqs-dbload2 /opt/colabfold/bin/
COPY --from=mmseqs /opt/mmseqs/build.txt /opt/colabfold/bin/mmseqs-build.txt
COPY --from=mmseqs /opt/mmseqs/source.txt /opt/colabfold/bin/mmseqs-source.txt

# ------------------------------------------------------------------------------------------------ 7. entry point and checks
# provenance, shown by `colabfold version` (the CI build passes the repository and the commit)
ARG SOURCE_URL=https://github.com/gbouras13/setonix-ports
ARG SOURCE_REV=unknown
COPY container/colabfold container/colabfold_check.py /opt/colabfold/container/
COPY tests/ /opt/colabfold/tests/
RUN chmod 0755 /opt/colabfold/scripts/bin/nvidia-smi /opt/colabfold/scripts/*.sh /opt/colabfold/bin/mmseqs /opt/colabfold/bin/mmseqs-dbload2 \
 && install -m 0755 /opt/colabfold/container/colabfold /usr/local/bin/colabfold \
 && echo "source $SOURCE_URL @ $SOURCE_REV" > /opt/colabfold/source.txt \
 && chmod -R a+rX /opt/colabfold /opt/python \
 && colabfold check --no-gpu \
 && colabfold help > /dev/null \
 && colabfold mmseqs version

ENV PATH=/opt/colabfold/scripts/bin:/opt/colabfold/bin:/opt/colabfold/venv/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONUNBUFFERED=1 CFS_W=/opt/colabfold
LABEL org.opencontainers.image.title="colabfold-setonix" \
      org.opencontainers.image.description="ColabFold 1.6.3 (AlphaFold2-Multimer v3) with the ColabFold kit's modes and colabfold_search on MMseqs2's GPU build, for AMD MI250X (gfx90a), ROCm 7.2.4, jax 0.10.2. No weights inside." \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.revision="${SOURCE_REV}" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT AND GPL-3.0-only"
WORKDIR /opt/colabfold
ENTRYPOINT ["/usr/local/bin/colabfold"]
CMD ["help"]
