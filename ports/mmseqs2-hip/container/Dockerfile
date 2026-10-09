# syntax=docker/dockerfile:1.7
# MMseqs2 with its HIP/ROCm GPU backend, for Pawsey Setonix (AMD MI250X GCDs, gfx90a). Upstream master at the commit the Setonix build
# validated (564f40d8, 2026-09-30), compiled the way af3-setonix/mmseqs2-hip/BUILD.md compiles it on Setonix:
#   - GCC 14 and ROCm 7.2.4;
#   - gfx90a only, AVX2 (Zen 3);
#   - dynamic GNU OpenMP;
#   - an RPATH to the ROCm libraries.
# Here the ROCm libraries are AMD's 7.2.4 packages, not Pawsey's module. That build's GPU search reproduced MMseqs2's CPU ungapped
# prefilter exactly (BUILD.md, job 50232402); slurm/test_gpu.sbatch checks this image the same way, and against the native binary.
#
# Two stages: the compiler image (AMD's ROCm 7.2.4 development image) builds the binary; the runtime image is Ubuntu 24.04 with only
# the HIP runtime from AMD's 7.2.4 apt repository. Images cannot be built on Setonix; GitHub Actions builds this one
# (.github/workflows/mmseqs2-hip-image.yml). Run it on Setonix WITHOUT --rocm (that binds the host's ROCm 6.3.0 libraries, same sonames).

ARG BUILD_IMAGE=rocm/dev-ubuntu-24.04:7.2.4@sha256:bdc8e61026cbb844ede93d44d2c50055f51ebb2041906b60182bf3bee3139054
ARG RUNTIME_IMAGE=ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55

# ------------------------------------------------------------------------------------------------ build
FROM ${BUILD_IMAGE} AS build
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8
# hipCUB / rocPRIM / rocThrust headers for libmarv (upstream's ROCm image installs the same), GCC 14 (the native build's compiler),
# git, bzip2 / zlib headers, and Python for CMake 3.31 + Ninja from pip (the native build's versions)
RUN apt-get update --quiet \
 && apt-get install --yes --quiet --no-install-recommends \
      hipcc hip-dev hipcub-dev rocprim-dev rocthrust-dev g++-14 git curl ca-certificates zlib1g-dev libbz2-dev python3-venv \
 && rm -rf /var/lib/apt/lists/*
# Rust for the block-aligner crate: the stable release of the native build's day (2026-10-01)
ARG RUST_VERSION=1.99.0
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup.sh \
 && RUSTUP_HOME=/opt/rust CARGO_HOME=/opt/rust sh /tmp/rustup.sh -y --no-modify-path --profile minimal --default-toolchain "$RUST_VERSION" \
 && rm -f /tmp/rustup.sh \
 && python3 -m venv /opt/tools && /opt/tools/bin/pip install --no-cache-dir "cmake>=3.31,<3.32" ninja
ARG MMSEQS_REPO=https://github.com/soedinglab/MMseqs2.git
ARG MMSEQS_COMMIT=564f40d8857f4eca4e1dfe100c67c155b1933e70
RUN git clone --quiet --no-checkout "$MMSEQS_REPO" /src/MMseqs2 \
 && git -C /src/MMseqs2 -c advice.detachedHead=false checkout --quiet "$MMSEQS_COMMIT" \
 && test "$(git -C /src/MMseqs2 rev-parse HEAD)" = "$MMSEQS_COMMIT"
# BUILD.md's CMake call. --gcc-install-dir pins ROCm's clang (which compiles the GPU code and links) to GCC 14's libstdc++; the OpenMP
# settings keep the binary on GCC's libgomp although clang links; HAVE_AVX2 also turns off -march=native.
ARG HIP_ARCH=gfx90a
RUN export ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm HIP_PLATFORM=amd RUSTUP_HOME=/opt/rust CARGO_HOME=/opt/rust CC=gcc-14 CXX=g++-14 \
 && export PATH=/opt/tools/bin:/opt/rust/bin:$PATH \
 && GCC_DIR=$(dirname "$(g++-14 -print-libgcc-file-name)") \
 && cmake -S /src/MMseqs2 -B /src/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/mmseqs \
      -DHAVE_AVX2=1 -DENABLE_HIP=1 -DCMAKE_HIP_ARCHITECTURES="$HIP_ARCH" \
      -DCMAKE_HIP_COMPILER=$ROCM_PATH/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$ROCM_PATH \
      -DCMAKE_HIP_FLAGS="--rocm-path=$ROCM_PATH --gcc-install-dir=$GCC_DIR" \
      -DCMAKE_PREFIX_PATH=$ROCM_PATH \
      -DOpenMP_C_FLAGS=-fopenmp -DOpenMP_C_LIB_NAMES=gomp \
      -DOpenMP_CXX_FLAGS=-fopenmp -DOpenMP_CXX_LIB_NAMES=gomp -DOpenMP_gomp_LIBRARY="$GCC_DIR/libgomp.so" \
      -DCMAKE_INSTALL_RPATH=$ROCM_PATH/lib -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
 && cmake --build /src/build -j"$(nproc --all)" \
 && cmake --install /src/build > /dev/null \
 && B=/opt/mmseqs/bin/mmseqs \
 && { echo "MMseqs2 $MMSEQS_REPO @ $MMSEQS_COMMIT ($(git -C /src/MMseqs2 log -1 --format=%cI))"; \
      echo "built for $HIP_ARCH with $(g++-14 --version | head -1); $($ROCM_PATH/lib/llvm/bin/clang++ --version | head -1)"; \
      echo "rocm-core $(dpkg-query -W -f '${Version}' rocm-core); $(cmake --version | head -1); $(/opt/rust/bin/rustc --version)"; \
      echo "GPU code objects: $($ROCM_PATH/bin/roc-obj-ls "$B" 2>/dev/null | awk '{print $2}' | sort -u | tr '\n' ' ')"; } \
      > /opt/mmseqs/build.txt \
 && cat /opt/mmseqs/build.txt && "$B" version \
 && test "$(grep -a -c "amdgcn-amd-amdhsa--$HIP_ARCH" "$B")" -gt 0 \
 && { ! ldd "$B" | grep "not found"; }

# ------------------------------------------------------------------------------------------------ runtime
FROM ${RUNTIME_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8
# the HIP runtime of AMD's ROCm 7.2.4 apt repository (what the binary links: libamdhip64, libhsa-runtime64, libamd_comgr), GNU OpenMP and
# libatomic (GCC's runtime libraries the binary links dynamically, as natively)
ARG ROCM_VERSION=7.2.4
RUN apt-get update --quiet \
 && apt-get install --yes --quiet --no-install-recommends ca-certificates curl gpg libgomp1 libatomic1 libbz2-1.0 zlib1g gawk \
 && mkdir -p /etc/apt/keyrings \
 && curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor > /etc/apt/keyrings/rocm.gpg \
 && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/$ROCM_VERSION noble main" \
      > /etc/apt/sources.list.d/rocm.list \
 && printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' > /etc/apt/preferences.d/rocm-pin-600 \
 && apt-get update --quiet \
 && apt-get install --yes --quiet --no-install-recommends hip-runtime-amd hsa-rocr comgr rocm-core \
 && apt-get purge --yes --auto-remove gpg \
 && rm -rf /var/lib/apt/lists/* \
 && v=$(dpkg-query -W -f '${Version}' rocm-core) \
 && case "$v" in "${ROCM_VERSION}".*) echo "ROCm runtime: rocm-core $v";; *) echo "rocm-core is $v, not ${ROCM_VERSION}" >&2; exit 1;; esac
COPY --from=build /opt/mmseqs/ /opt/mmseqs/
COPY scripts/mmseqs-dbload2 /opt/mmseqs/bin/
COPY tests/ /opt/mmseqs/tests/
COPY container/mmseqs /usr/local/bin/mmseqs
ARG SOURCE_URL=https://github.com/gbouras13/setonix-ports
ARG SOURCE_REV=unknown
# Build-time checks (no GPU here): every library resolves inside the image, the GPU options are compiled in, and a CPU search of the
# 246 test proteins against themselves finds their self-hits.
RUN chmod 0755 /usr/local/bin/mmseqs /opt/mmseqs/bin/mmseqs-dbload2 \
 && echo "source $SOURCE_URL @ $SOURCE_REV" > /opt/mmseqs/source.txt \
 && { [ -e /opt/rocm ] || ln -s "/opt/rocm-$ROCM_VERSION" /opt/rocm; } \
 && test -e /opt/rocm/lib/libamdhip64.so.7 \
 && { ! LD_LIBRARY_PATH=/opt/rocm/lib ldd /opt/mmseqs/bin/mmseqs | grep "not found"; } \
 && mmseqs version \
 && mmseqs search -h 2>&1 | grep -q -- "--gpu-server" \
 && mmseqs createdb /opt/mmseqs/tests/SAOMS1_phanotate.faa /tmp/q > /dev/null \
 && mmseqs search /tmp/q /tmp/q /tmp/r /tmp/t --threads 2 > /dev/null \
 && mmseqs convertalis /tmp/q /tmp/q /tmp/r /tmp/r.m8 > /dev/null \
 && test "$(awk '$1 == $2' /tmp/r.m8 | cut -f1 | sort -u | wc -l)" -ge 240 \
 && rm -rf /tmp/q* /tmp/r* /tmp/t
ENV PATH=/usr/local/bin:/opt/mmseqs/bin:/usr/sbin:/usr/bin:/sbin:/bin
LABEL org.opencontainers.image.title="mmseqs2-hip-setonix" \
      org.opencontainers.image.description="MMseqs2 (master 564f40d8) with its HIP GPU backend for AMD MI250X (gfx90a), ROCm 7.2.4." \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.revision="${SOURCE_REV}" \
      org.opencontainers.image.licenses="GPL-3.0-only"
ENTRYPOINT ["/usr/local/bin/mmseqs"]
