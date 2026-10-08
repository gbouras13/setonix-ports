# syntax=docker/dockerfile:1.7
# af3_jax for Pawsey Setonix (AMD Instinct MI250X, gfx90a): stock AlphaFold 3 in JAX (the sokrypton/alphafold3 v3.1.4 fork that the
# af3_jax kit pins) on ROCm 7.2.4 + jax / jaxlib / jax-rocm7-plugin / jax-rocm7-pjrt 0.10.2, with the port's validated defaults.
#
# The image IS an af3_jax tree at /opt/af3jax -- stockenv/ (the Python environment), alphafold/ (the fork), ubm/ (the kit), scripts/
# (the port's wrapper and launchers), weights/ (bind your weights here or pass --model_dir) -- and its entry point runs the same
# wrapper the native install runs (scripts/af3jax_predict.sh, v13: setups A / B / C chosen by size, library GEMMs with their caps, the
# pair-logits lever, copy-only serialisation, compile cache). So native and container runs share one code path and one set of checks.
#
# Inside: ROCm 7.2.4 user space from AMD's image and apt repository; Python 3.12.1 (python-build-standalone) and uv 0.9.24, the kit's
# pins; the kit's requirements.lock with its 15 CUDA lines swapped for the jax-rocm7 plugin of the same jax release; the fork at the
# kit's pin with the kit's one declared patch, built as upstream builds it (or a wheel you supply, WHEELS_FROM); the two ROCm patches
# applied in place with their --check. NOT inside: any weights or parameters (the AlphaFold 3 parameters' terms forbid redistribution:
# bind your own copy), inputs, outputs, genetic databases.
#
# Run on Setonix WITHOUT --rocm: that flag binds the host's ROCm 6.3.0 libraries over the image's (same sonames). `af3jax check` fails
# loudly if any GPU library was loaded from outside the image. Images cannot be built on Setonix; GitHub Actions builds this one
# (.github/workflows/af3jax-image.yml). Context: ports/af3jax/.

ARG BASE_IMAGE=rocm/dev-ubuntu-24.04:7.2.4@sha256:bdc8e61026cbb844ede93d44d2c50055f51ebb2041906b60182bf3bee3139054
FROM ${BASE_IMAGE}

# Each build argument is declared just before the step that uses it: every RUN after an ARG sees it in its environment, so a changed
# value (SOURCE_REV changes with every commit) re-runs only the steps from its declaration on, and the CI build's layer cache holds.

ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 PIP_NO_CACHE_DIR=1

# ------------------------------------------------------------------------------------------------ 1a. ROCm's big GPU math libraries, pruned
# hipBLASLt, rocBLAS and MIOpen install about 8 GiB, mostly kernels for other GPUs. Only PRUNE_GPU_ARCH's kernel files are kept, pruned
# IN THE SAME LAYER as the install, so the image shrinks and not just the final filesystem. Setonix is gfx90a; empty keeps every
# architecture. The ROCm install is split into two layers so that each stays well inside ghcr.io's limits (10 GB per layer, 10 minutes
# per upload).
ARG ROCM_VERSION=7.2.4
ARG PRUNE_GPU_ARCH=gfx90a
COPY container/prune_rocm_arch.py /opt/af3jax/container/
RUN apt-get update --quiet \
 && apt-get install --yes --quiet --no-install-recommends hipblaslt rocblas hipblas miopen-hip python3 \
 && rm -rf /var/lib/apt/lists/* \
 && v=$(dpkg-query -W -f '${Version}' rocm-core) \
 && case "$v" in "${ROCM_VERSION}".*) echo "ROCm user space: rocm-core $v";; *) echo "rocm-core is $v, not ${ROCM_VERSION}" >&2; exit 1;; esac \
 && if [ -n "$PRUNE_GPU_ARCH" ]; then \
      python3 /opt/af3jax/container/prune_rocm_arch.py --keep "$PRUNE_GPU_ARCH" \
        /opt/rocm/lib/hipblaslt/library /opt/rocm/lib/rocblas/library /opt/rocm/share/miopen/db /opt/rocm/lib/miopen \
      && echo "pruned to $PRUNE_GPU_ARCH: hipBLASLt, rocBLAS, MIOpen kernel files" > /opt/af3jax/rocm-pruned.txt ; \
    fi

# ------------------------------------------------------------------------------------------------ 1b. the other ROCm libraries + build tools
# What XLA's ROCm plugin needs beyond the base image's HIP runtime (its shared objects link libamdhip64.so.7, libhipsparse.so.4,
# libMIOpen.so.1; XLA opens hipBLASLt, rocBLAS, hipBLAS, rocSOLVER/hipSOLVER, rocFFT/hipFFT, RCCL and roctracer at run time), the ROCm
# device bitcode and ld.lld XLA compiles kernels with, rocm-smi / rocprofv3 for diagnosis, and upstream AlphaFold 3's build tools.
# libdw1t64 (elfutils' libdw.so.1): the plugin needs it, and the ROCm packages do not pull it in (Setonix hosts have it in the OS).
RUN apt-get update --quiet \
 && apt-get install --yes --quiet --no-install-recommends \
      rccl hipfft rocfft hipsparse rocsparse hipsolver rocsolver hiprand rocrand roctracer \
      rocm-llvm rocm-device-libs rocm-smi-lib rocprofiler-sdk libdw1t64 \
      git wget curl ca-certificates gcc g++ make zlib1g-dev zstd patch \
 && rm -rf /var/lib/apt/lists/* \
 && dpkg-query -W -f '${Package} ${Version}\n' rocm-core hip-runtime-amd hsa-rocr comgr hipblaslt rocblas miopen-hip rccl rocfft rocsolver rocsparse \
      > /opt/af3jax/rocm-packages.txt \
 && { cat /opt/af3jax/rocm-pruned.txt >> /opt/af3jax/rocm-packages.txt 2>/dev/null || true; }

# ------------------------------------------------------------------------------------------------ 2. Python 3.12.1 + uv (the kit's pins)
RUN curl -fsSL -o /tmp/python.tar.gz \
      "https://github.com/indygreg/python-build-standalone/releases/download/20240107/cpython-3.12.1+20240107-x86_64_v3-unknown-linux-gnu-install_only.tar.gz" \
 && echo "a7bcc6c9f66dbd47ea99615f30f101c5d2dd0084ca333d2f7336e64050951338  /tmp/python.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/python.tar.gz -C /tmp && mv /tmp/python /opt/python && rm -f /tmp/python.tar.gz \
 && /opt/python/bin/python3.12 -V && /opt/python/bin/python3.12 -m pip install --no-cache-dir uv==0.9.24

# ------------------------------------------------------------------------------------------------ 3. the kit, at its pin (sparse)
# the af3_jax kit (public, Apache-2.0), sparse-checked-out at the commit the Setonix port ran
ARG KIT_REPO=https://github.com/anthropics/uplifting-biomolecular-modeling.git
ARG KIT_COMMIT=f4f62fa6592ae4938d49b1757bea0cfeff9f468e
RUN git clone --quiet --filter=blob:none --no-checkout "$KIT_REPO" /opt/af3jax/ubm \
 && git -C /opt/af3jax/ubm sparse-checkout set --cone af3_jax common/opt_core \
 && git -C /opt/af3jax/ubm fetch --quiet --filter=blob:none origin "$KIT_COMMIT" \
 && git -C /opt/af3jax/ubm -c advice.detachedHead=false checkout --quiet "$KIT_COMMIT" \
 && test "$(git -C /opt/af3jax/ubm rev-parse HEAD)" = "$KIT_COMMIT" \
 && rm -rf /opt/af3jax/ubm/.git \
 && echo "kit $KIT_REPO @ $KIT_COMMIT" > /opt/af3jax/kit.txt

# ------------------------------------------------------------------------------------------------ 4. the stock environment
# The kit's lock minus its 15 CUDA lines plus the two ROCm lines of the same jax release (scripts/setup-stock-stack.sh's derivation).
RUN L=/opt/af3jax/ubm/af3_jax/environment/requirements.lock \
 && n=$(grep -cE '^(nvidia-[a-z0-9-]+-cu12|jax-cuda12-(pjrt|plugin))==' "$L") \
 && [ "$n" = 15 ] || { echo "expected 15 CUDA lines in $L, found $n" >&2; exit 1; } \
 && jv=$(sed -n 's/^jax==//p' "$L") && [ -n "$jv" ] \
 && grep -vE '^(nvidia-[a-z0-9-]+-cu12|jax-cuda12-(pjrt|plugin))==' "$L" > /opt/af3jax/requirements-rocm.lock \
 && printf 'jax-rocm7-plugin==%s\njax-rocm7-pjrt==%s\n' "$jv" "$jv" >> /opt/af3jax/requirements-rocm.lock \
 && /opt/python/bin/python3.12 -m uv venv --python /opt/python/bin/python3.12 /opt/af3jax/stockenv \
 && /opt/python/bin/python3.12 -m uv pip install --python /opt/af3jax/stockenv/bin/python --no-cache --no-deps -r /opt/af3jax/requirements-rocm.lock

# ------------------------------------------------------------------------------------------------ 5. AlphaFold 3: the fork + its wheel
# The fork at the kit's pin with its one declared patch, at /opt/af3jax/alphafold (a real directory: the wrapper copies it to TMPDIR
# and patches the copy), linked from upstream's /app/alphafold. The wheel: yours when container/wheels/ holds it (sha256 checked), else
# compiled by upstream's own build (abseil / pybind11 / libcifpp / dssp from GitHub, the wwPDB dictionary of the build day). Then
# build_data. (container/wheels/ always holds its README, so this COPY never matches nothing, and editing this file never re-runs it.)
# WHEELS_FROM: auto = container/wheels/alphafold3_open-3.1.4-cp312-cp312-linux_x86_64.whl when present (its sha256 must then be
# STOCK_WHEEL_SHA256, the wheel validated on Setonix on 2026-09-22), else compiled from the pinned source (needs network); prebuilt;
# build. BUILD_JOBS=0 uses every core.
ARG WHEELS_FROM=auto
ARG STOCK_WHEEL_SHA256=6be3b1c4495ae729a9f3663740323f89488d8e12c4d82b95e7ee5f4c954700cc
ARG BUILD_JOBS=0
COPY container/wheels/ /tmp/wheels/
RUN bash /opt/af3jax/ubm/af3_jax/stock/unpack_src.sh /opt/af3jax/alphafold \
 && mkdir -p /app && ln -s /opt/af3jax/alphafold /app/alphafold \
 && W=/tmp/wheels/alphafold3_open-3.1.4-cp312-cp312-linux_x86_64.whl; route="$WHEELS_FROM"; \
    if [ "$route" = auto ]; then if [ -f "$W" ]; then route=prebuilt; else route=build; fi; fi; \
    case "$route" in \
      prebuilt) echo "$STOCK_WHEEL_SHA256  $W" | sha256sum -c - ;; \
      build) J=$BUILD_JOBS; [ "$J" -gt 0 ] || J=$(nproc --all); export MAX_JOBS=$J CMAKE_BUILD_PARALLEL_LEVEL=$J MAKEFLAGS=-j$J CFLAGS=-g0 CC=gcc CXX=g++; \
             rm -f "$W"; cp -a /opt/af3jax/alphafold /tmp/af3src \
             && /opt/python/bin/python3.12 -m uv build --wheel --python /opt/python/bin/python3.12 --out-dir /tmp/wheels /tmp/af3src \
             && rm -rf /tmp/af3src /root/.cache ;; \
      *) echo "WHEELS_FROM=$route: auto | prebuilt | build" >&2; exit 2 ;; \
    esac \
 && /opt/python/bin/python3.12 -m uv pip install --python /opt/af3jax/stockenv/bin/python --no-cache --no-deps "$W" \
 && /opt/af3jax/stockenv/bin/build_data \
 && echo "alphafold3 wheel: route=$route sha256=$(sha256sum "$W" | cut -c1-64)" > /opt/af3jax/wheel.txt \
 && cat /opt/af3jax/wheel.txt && rm -rf /tmp/wheels

# ------------------------------------------------------------------------------------------------ 6. HMMER 3.4 (data pipeline only)
# HMMER 3.4 with upstream's jackhmmer patch, for the data pipeline (inputs that already carry MSAs do not need it); 0 leaves it out
ARG WITH_HMMER=1
RUN if [ "$WITH_HMMER" = 1 ]; then \
      P=/opt/af3jax/alphafold/docker/jackhmmer_seq_limit.patch; [ -f "$P" ] || { echo "no $P in the fork" >&2; exit 1; }; \
      mkdir -p /tmp/hmmer && cd /tmp/hmmer \
      && wget -q http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz \
      && echo "ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3  hmmer-3.4.tar.gz" | sha256sum -c - \
      && tar -xzf hmmer-3.4.tar.gz && cp "$P" . && patch -p0 < jackhmmer_seq_limit.patch \
      && cd hmmer-3.4 && ./configure --prefix /hmmer >/dev/null && make -j"$(nproc --all)" >/dev/null && make install >/dev/null \
      && (cd easel && make install >/dev/null) && cd / && rm -rf /tmp/hmmer && /hmmer/bin/jackhmmer -h | head -2; \
    fi

# ------------------------------------------------------------------------------------------------ 7. the port: wrapper, launchers, checks
# The ROCm patches are applied in place AND keep their .rocm-orig originals: the wrapper copies alphafold/ and tokamax to TMPDIR,
# restores the originals there and re-patches its copies, exactly as it does natively, so the read-only image is never written.
# provenance, shown by `af3jax version` (the CI build passes the repository and the commit)
ARG SOURCE_URL=https://github.com/gbouras13/setonix-ports
ARG SOURCE_REV=unknown
COPY scripts/ /opt/af3jax/scripts/
COPY cases/ /opt/af3jax/cases/
COPY container/af3jax container/af3jax_check.py /opt/af3jax/container/
RUN PY=/opt/af3jax/stockenv/bin/python \
 && SITE=$($PY -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') \
 && test "$SITE" = /opt/af3jax/stockenv/lib/python3.12/site-packages \
 && $PY /opt/af3jax/scripts/patch_rocm_capability.py /opt/af3jax/alphafold && $PY /opt/af3jax/scripts/patch_rocm_capability.py /opt/af3jax/alphafold --check \
 && $PY /opt/af3jax/scripts/patch_tokamax_rocm.py "$SITE" && $PY /opt/af3jax/scripts/patch_tokamax_rocm.py "$SITE" --check \
 && mkdir -p /opt/af3jax/weights/p2 \
 && chmod 0755 /opt/af3jax/scripts/af3jax_predict.sh \
 && install -m 0755 /opt/af3jax/container/af3jax /usr/local/bin/af3jax \
 && echo "source $SOURCE_URL @ $SOURCE_REV" > /opt/af3jax/source.txt \
 && chmod -R a+rX /opt/af3jax /opt/python \
 && af3jax check --no-gpu \
 && AF3JAX_W=/opt/af3jax ROCM_PATH=/opt/rocm bash /opt/af3jax/scripts/af3jax_predict.sh --help >/dev/null

ENV PATH=/opt/af3jax/stockenv/bin:/hmmer/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONUNBUFFERED=1 AF3JAX_W=/opt/af3jax
LABEL org.opencontainers.image.title="af3jax-setonix" \
      org.opencontainers.image.description="Stock AlphaFold 3 (JAX, sokrypton v3.1.4 fork) with the Setonix port's wrapper for AMD MI250X (gfx90a), ROCm 7.2.4, jax 0.10.2. No weights inside." \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.revision="${SOURCE_REV}" \
      org.opencontainers.image.licenses="Apache-2.0 AND LicenseRef-AlphaFold3"
WORKDIR /opt/af3jax
ENTRYPOINT ["/usr/local/bin/af3jax"]
CMD ["help"]
