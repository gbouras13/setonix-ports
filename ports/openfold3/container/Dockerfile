# syntax=docker/dockerfile:1.7
# OpenFold3 0.5.0 for Pawsey Setonix (AMD Instinct MI250X, gfx90a), row-sharded over 2, 4 or 8 GCDs of one node through the
# optimisation kit's tp_rowpair line, with the port's gfx90a defaults (fp16 fused triangle attention and the rest; the evidence is in
# af3-setonix/THROUGHPUT.md and RUNTIME.md). The image IS the native port's tree, rebuilt at /opt/openfold3 in the native layout,
# because the native launcher (scripts/predict-sharded.sh, verbatim) finds everything under one root:
#   py312/        the environment (a venv here, conda natively; the launcher puts py312/bin first on PATH): torch 2.14.0+rocm7.2 and
#                 triton-rocm 3.8.0 (download.pytorch.org/whl/rocm7.2) and the rest of the native process's packages, pinned exactly by
#                 environment/requirements.lock and installed --no-deps; openfold3, opt_core and openfold3_opt editable, as natively;
#   openfold3/    OpenFold3 0.5.0 (aqlaboratory/openfold-3, tag v0.5.0), with the port's two seed pins and the tunable Triton constants;
#   of3_opt_src/opt, opt_core_src/opt_core
#                 the optimisation kit's OpenFold3 adapter and its shared core, at the kit's commit, with the port's patches;
#   kit/          the port's launcher scripts and runner yamls, verbatim; patches/, tools/, environment/: the rest of this directory.
# NO WEIGHTS: `openfold3 fetch-weights DIR` downloads the OpenBind checkpoint (2.29 GB) through OpenFold3's own downloader, from its
# public bucket, and checks it against the S3 object's own checksums.
#
# THE SOURCES ARE THE NATIVE PORT'S, BYTE FOR BYTE. The native trees were copied from the kit at f4f62fa and patched in place on
# Setonix; that tree is now being taken by the /scratch purge. Step 3 rebuilds it from public pins and the patch scripts (patches/,
# from the native tree, in the order they were applied), then checks every file of the two kit trees and the three patched OpenFold3
# files against environment/native-sources.sha256, the native tree's own digests (2026-10-21). One file of the native tree is not
# carried (a leftover backup, fpf_trimul/arch_tables.json.retune_backup), and the kit's prebuilt CUDA binaries, which the native copy
# never had and which cannot load on gfx90a, are deleted (environment/cuda-prebuilt.txt: 249 files and two directories of blobs and
# test vectors, 254 MB). Any other difference fails the build.
#
# Run on Setonix WITHOUT --rocm (it binds the host's ROCm 6.3.0 libraries, same sonames). `openfold3 check` fails if any GPU library
# was loaded from outside the image. Images cannot be built on Setonix; GitHub Actions builds this one (.github/workflows/openfold3-image.yml).
# Context: ports/openfold3/.

ARG BASE_IMAGE=rocm/dev-ubuntu-24.04:7.2.4@sha256:bdc8e61026cbb844ede93d44d2c50055f51ebb2041906b60182bf3bee3139054
FROM ${BASE_IMAGE}
# Each build argument is declared just before the step that uses it: every RUN after an ARG sees it in its environment, so a changed
# value (SOURCE_REV changes with every commit) re-runs only the steps from its declaration on, and the CI build's layer cache holds.
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# ------------------------------------------------------------------------------------------------ 1. tools
# No ROCm math libraries are installed: the torch 2.14.0+rocm7.2 wheel carries its own complete ROCm 7.2 runtime in torch/lib, and that
# is the one the native port runs on too. git and curl fetch the sources and the interpreter; python3 runs the prune script.
# build-essential is for Triton (it compiles its kernel launcher with the system C compiler when it first builds a kernel: the fused
# triangle attention, fpf_v4, the LayerNorm kernel) and for func-timeout, the lock's one source distribution.
RUN mkdir -p /opt/openfold3 \
 && apt-get update --quiet \
 && apt-get install --yes --quiet --no-install-recommends git curl ca-certificates python3 build-essential \
 && rm -rf /var/lib/apt/lists/* \
 && dpkg-query -W -f '${Package} ${Version}\n' rocm-core hip-runtime-amd hsa-rocr comgr \
      > /opt/openfold3/rocm-packages.txt

# ------------------------------------------------------------------------------------------------ 2. Python 3.12
# The native env is conda-forge's CPython 3.12.14, on /scratch. This is the same minor version from python-build-standalone (the
# protenix, boltz2 and opendde images' build); x86_64_v3 covers Setonix's Trento (Zen 3). Nothing in the port's numerics depends on the
# interpreter's patch level -- the validation job proves that by comparing the image's outputs with the native port's, byte for byte.
ARG PYTHON_URL=https://github.com/astral-sh/python-build-standalone/releases/download/20260211/cpython-3.12.12%2B20260211-x86_64_v3-unknown-linux-gnu-install_only.tar.gz
ARG PYTHON_SHA256=e10e333796fab1a7793171ea5c2b8cadc6f9c5dfdcc06815fa402c05cbd76d81
RUN curl -fsSL -o /tmp/python.tar.gz "$PYTHON_URL" \
 && echo "$PYTHON_SHA256  /tmp/python.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/python.tar.gz -C /tmp && mv /tmp/python /opt/python && rm -f /tmp/python.tar.gz \
 && /opt/python/bin/python3.12 -V

# ------------------------------------------------------------------------------------------------ 3. the sources, rebuilt and checked
# OpenFold3 at its release commit (the native tree was `git clone --branch v0.5.0`), the kit at its pin, sparse: its OpenFold3 adapter
# (openfold3/opt, written for OpenFold3 0.4.1; port_of3_050.py ports it to 0.5.0) and the shared core (common/opt_core). Then every patch,
# in the order the native port applied them (make_tunable_gfx90a first, 2026-09-18; patch_ln_kernel last, 2026-10-08), each against its
# in-image path. Then the digests: the build stops on any file that is not the native one.
ARG OPENFOLD3_REPO=https://github.com/aqlaboratory/openfold-3.git
ARG OPENFOLD3_TAG=v0.5.0
ARG OPENFOLD3_COMMIT=c4771653c5d0a3ebb0b3af71b05efd64bc44ee86
ARG KIT_REPO=https://github.com/anthropics/uplifting-biomolecular-modeling.git
ARG KIT_COMMIT=f4f62fa6592ae4938d49b1757bea0cfeff9f468e
COPY patches/ /opt/openfold3/patches/
COPY environment/native-sources.sha256 environment/cuda-prebuilt.txt /opt/openfold3/environment/
RUN W=/opt/openfold3 \
 && git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$OPENFOLD3_TAG" "$OPENFOLD3_REPO" $W/openfold3 \
 && test "$(git -C $W/openfold3 rev-parse HEAD)" = "$OPENFOLD3_COMMIT" \
 && rm -rf $W/openfold3/.git \
 && echo "openfold3 $OPENFOLD3_REPO @ $OPENFOLD3_TAG ($OPENFOLD3_COMMIT)" > $W/openfold3.txt \
 && git clone --quiet --filter=blob:none --no-checkout "$KIT_REPO" /tmp/kit \
 && git -C /tmp/kit sparse-checkout set --cone openfold3 common \
 && git -C /tmp/kit fetch --quiet --filter=blob:none origin "$KIT_COMMIT" \
 && git -C /tmp/kit -c advice.detachedHead=false checkout --quiet "$KIT_COMMIT" \
 && test "$(git -C /tmp/kit rev-parse HEAD)" = "$KIT_COMMIT" \
 && mkdir -p $W/of3_opt_src $W/opt_core_src $W/licenses/kit-openfold3 \
 && cp -a /tmp/kit/openfold3/opt $W/of3_opt_src/opt \
 && cp -a /tmp/kit/common/opt_core $W/opt_core_src/opt_core \
 && cp -a /tmp/kit/openfold3/LICENSE /tmp/kit/openfold3/THIRD_PARTY_NOTICES.md /tmp/kit/openfold3/third_party_licenses \
          $W/licenses/kit-openfold3/ \
 && rm -rf /tmp/kit \
 && echo "kit $KIT_REPO @ $KIT_COMMIT (openfold3/opt -> of3_opt_src/opt, common/opt_core -> opt_core_src/opt_core)" > $W/kit.txt \
 && (cd $W/opt_core_src/opt_core && xargs -d '\n' rm -rf -- < $W/environment/cuda-prebuilt.txt) \
 && O=$W/of3_opt_src/opt/openfold3_opt C=$W/opt_core_src/opt_core/opt_core F=$W/openfold3/openfold3 \
 && P=$W/patches PY=/opt/python/bin/python3.12 \
 && $PY -I $P/make_tunable_gfx90a.py --apply $W/openfold3 \
 && $PY -I $P/port_of3_050.py --apply $O/tp_rowpair \
 && $PY -I $P/patch_templ_policy.py --apply $O/report.py \
 && $PY -I $P/patch_serve_fp32.py --apply $O/tp_rowpair/pairstack.py \
 && $PY -I $P/patch_conformer_seed.py --apply $F/core/data/primitives/structure/conformer.py \
 && $PY -I $P/patch_refpos_seed.py --apply $F/core/data/pipelines/featurization/conformer.py \
 && $PY -I $P/patch_perkey_digest.py --apply $O/tp_rowpair/model.py \
 && $PY -I $P/patch_qblock_tail.py --apply $C/mem/rowpair/triatt.py \
 && $PY -I $P/patch_flash_ip.py --apply $C/kernels/flash_triattn.py \
 && $PY -I $P/patch_triatt_fp16.py --apply $O/tp_rowpair/pairstack.py \
 && $PY -I $P/patch_triatt_diag.py --apply $O/tp_rowpair/pairstack.py \
 && $PY -I $P/patch_triatt_autocast.py --apply $O/tp_rowpair/pairstack.py \
 && $PY -I $P/patch_trimul_hipblaslt.py --apply $C/mem/rowpair/trimul.py \
 && $PY -I $P/patch_sample_chunk.py --apply $O/sample_loop.py \
 && $PY -I $P/patch_dit_ln_skip.py --apply $O/tp_rowpair/diffusion.py \
 && $PY -I $P/patch_atom_pair_cache.py --apply $O/tp_rowpair/diffusion.py \
 && $PY -I $P/patch_transpose_peers.py --apply $C/mem/rowpair/pairstack.py \
 && $PY -I $P/patch_ln_kernel.py --apply $O/tp_rowpair/model.py \
 && find $W -name __pycache__ -type d -prune -exec rm -rf {} + \
 && cd $W && sha256sum --check --quiet --strict environment/native-sources.sha256 \
 && awk '{print $2}' environment/native-sources.sha256 | grep -v '^openfold3/' | sort > /tmp/want.txt \
 && find of3_opt_src/opt opt_core_src/opt_core -type f | sort > /tmp/have.txt \
 && { cmp -s /tmp/want.txt /tmp/have.txt || { echo "the kit trees hold files the native tree does not, or lack some:"; \
      diff /tmp/want.txt /tmp/have.txt | head -20; exit 1; }; } \
 && rm -f /tmp/want.txt /tmp/have.txt \
 && echo "sources: $(wc -l < environment/native-sources.sha256) files = the native port's (environment/native-sources.sha256)"

# ------------------------------------------------------------------------------------------------ 4. the environment, pinned, pruned to gfx90a
# py312/: the launcher's name for the environment. setuptools and wheel first, at the native env's versions (conda's, there); then the
# lock's distributions (--no-deps: the native process's own versions, not openfold3's declared ranges); then the three source trees
# editable, as natively (`pip install -e`): openfold3 0.5.0, opt_core 0.5.228.0, openfold3_opt 0.8.20.1, whose build ships the kit's
# openfold3_opt_autoload.pth into site-packages (an import guard that stays off while OPENFOLD3_OPT is unset). openfold3's build needs
# setuptools-scm, which the native env does not hold either: its build gets pip's isolated build environment, as natively, and with
# no .git the version is its pyproject's fallback_version, 0.5.0. `pip check` is recorded, not enforced: the native env ran past
# openfold3's declared ranges.
#
# The prune: of the ~14 GiB in torch/, ~6 GiB are GPU kernels for cards that are not this one (hipBLASLt, rocBLAS, aotriton, MIOpen's
# databases); it runs in the same layer as the install, so the layer never holds the unpruned tree. The large .so files stay: they are
# DT_NEEDED entries of libtorch_hip.so.
ARG PRUNE_GPU_ARCH=gfx90a
ARG TORCH_INDEX=https://download.pytorch.org/whl/rocm7.2
COPY container/prune_rocm_arch.py /opt/openfold3/container/
COPY environment/requirements.lock /opt/openfold3/environment/
RUN W=/opt/openfold3 \
 && /opt/python/bin/python3.12 -m venv $W/py312 \
 && P=$W/py312/bin/python \
 && $P -m pip install --quiet --no-deps --no-compile setuptools==84.0.0 wheel==0.48.0 \
 && $P -m pip install --quiet --no-deps --no-compile --no-build-isolation \
      --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple -r $W/environment/requirements.lock \
 && $P -m pip install --quiet --no-deps --no-compile -e $W/openfold3 \
 && $P -m pip install --quiet --no-deps --no-compile --no-build-isolation -e $W/opt_core_src/opt_core -e $W/of3_opt_src/opt \
 && test "$($P -c 'import importlib.metadata as m; print(m.version("openfold3"), m.version("opt_core"), m.version("openfold3_opt"))')" \
       = "0.5.0 0.5.228.0 0.8.20.1" \
 && { $P -m pip check > $W/environment/pip-check.txt 2>&1 || true; } \
 && cat $W/environment/pip-check.txt \
 && $P -m pip freeze --all > $W/environment/image-freeze.txt \
 && L=$($P -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") \
 && test -f "$L/openfold3_opt_autoload.pth" \
 && if [ -n "$PRUNE_GPU_ARCH" ]; then \
      python3 $W/container/prune_rocm_arch.py --keep "$PRUNE_GPU_ARCH" \
        "$L/torch/lib/hipblaslt" "$L/torch/lib/rocblas" "$L/torch/lib/aotriton.images" "$L/torch/share/miopen" \
        | tee $W/rocm-pruned.txt ; \
    fi \
 && du -sh "$L/torch" "$L/triton" >> $W/rocm-pruned.txt \
 && find $W/openfold3 $W/of3_opt_src $W/opt_core_src -name __pycache__ -type d -prune -exec rm -rf {} +

# ------------------------------------------------------------------------------------------------ 5. the port: launcher, yamls, tools, entry point
# kit/scripts and kit/configs are the native kit's files, verbatim (the launcher's KIT_DIR default is $W/kit). tmp/ is a link to /tmp:
# the launcher puts its rank rendezvous file (ROWPAIR_STORE) at $W/tmp/rowpair-store-<job>-P<p>, and /tmp is the node's own disk inside
# Singularity. Everything else the launcher writes -- OUT, TMPDIR, the caches, the Triton cache -- the entry point points elsewhere.
# provenance, shown by `openfold3 version` (the CI build passes the repository and the commit)
ARG SOURCE_URL=https://github.com/gbouras13/setonix-ports
ARG SOURCE_REV=unknown
COPY scripts/ /opt/openfold3/kit/scripts/
COPY configs/ /opt/openfold3/kit/configs/
COPY tools/ /opt/openfold3/tools/
COPY tests/ /opt/openfold3/tests/
COPY container/openfold3 container/openfold3_check.py /opt/openfold3/container/
RUN W=/opt/openfold3 \
 && ln -s /tmp $W/tmp \
 && chmod 0755 $W/kit/scripts/*.sh \
 && install -m 0755 $W/container/openfold3 /usr/local/bin/openfold3 \
 && echo "source $SOURCE_URL @ $SOURCE_REV" > $W/source.txt \
 && chmod -R a+rX $W /opt/python \
 && openfold3 check --no-gpu \
 && bash $W/tests/test_entry.sh \
 && openfold3 help > /dev/null \
 && openfold3 version

# /usr/local/bin first. OpenFold3's own wheel installs `run_openfold` and `setup_openfold` into py312/bin, which the launcher needs on its
# PATH; nothing there is called openfold3, but the protenix and opendde images taught that an upstream CLI of the entry point's name on
# an earlier PATH entry shadows it for `singularity exec <sif> openfold3 ...` (jobs 51048476, 51048313). The check runs on the final PATH.
ENV PATH=/usr/local/bin:/opt/openfold3/py312/bin:/opt/rocm/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OF3_ROOT=/opt/openfold3
RUN test "$(command -v openfold3)" = /usr/local/bin/openfold3 \
 && test "$(command -v run_openfold)" = /opt/openfold3/py312/bin/run_openfold \
 && openfold3 version > /dev/null
LABEL org.opencontainers.image.title="openfold3-setonix" \
      org.opencontainers.image.description="OpenFold3 0.5.0 with the optimisation kit's row-sharded line on 2, 4 or 8 GCDs and the Setonix port's gfx90a defaults, on AMD MI250X (gfx90a), ROCm 7.2.4, torch 2.14.0. No weights: openfold3 fetch-weights." \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.revision="${SOURCE_REV}" \
      org.opencontainers.image.licenses="Apache-2.0"
WORKDIR /opt/openfold3
ENTRYPOINT ["/usr/local/bin/openfold3"]
CMD ["help"]
