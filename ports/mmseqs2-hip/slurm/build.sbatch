#!/bin/bash --login
#SBATCH --job-name=mmseqs2-hip-build
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=mmseqs2-hip-build.%j.out
#
# Build MMseqs2 (master, with the HIP/ROCm GPU backend) for the AMD MI250X (gfx90a) GPUs of Pawsey Setonix.
#
#   sbatch --account=<project> build.sbatch ROOT=/scratch/<project>/$USER/mmseqs2-hip [HIP_ARCH=gfx90a]
#   (arguments, not variables: Setonix sets SBATCH_EXPORT=NONE)
# This is a CPU-only job: the GPU code is compiled for gfx90a ahead of time, so no GPU is needed to build.
# Prepare these once on a login node (downloads only, see BUILD.md):
#   $ROOT/src   git clone https://github.com/soedinglab/MMseqs2.git
#   $ROOT/rust  Rust stable via rustup --profile minimal (RUSTUP_HOME and CARGO_HOME point here)
#   $ROOT/venv  python venv with "cmake>=3.31,<3.32" and ninja from pip
# Result: $ROOT/install/bin/mmseqs, with an RPATH to ROCm 7.2.4 so it runs without loading the module.
set -euo pipefail
for a in "$@"; do
  case "$a" in
    ROOT=*|HIP_ARCH=*) export "${a?}" ;;
    *) echo "unknown argument '$a': ROOT= HIP_ARCH="; exit 2 ;;
  esac
done
ROOT=${ROOT:?ROOT=<the directory with src/, rust/ and venv/ (BUILD.md, step 1)>}
ARCH=${HIP_ARCH:-gfx90a}                     # MI250X; add e.g. ";gfx942" for MI300 at the cost of build time
JOBS=${SLURM_CPUS_PER_TASK:-16}

# ROCm 7.2.4 from Pawsey's unsupported module tree (the same userspace the GPU nodes' torch/jax stacks use)
module use /software/setonix/unsupported
module load rocm/7.2.4
export ROCM_PATH=${ROCM_PATH:-/software/setonix/rocm/rocm-7.2.4}
export HIP_PATH=$ROCM_PATH HIP_PLATFORM=amd

# Rust (MMseqs2 builds the block-aligner crate through corrosion, offline) and cmake/ninja from the venv
export RUSTUP_HOME=$ROOT/rust/rustup CARGO_HOME=$ROOT/rust/cargo
export PATH=$ROOT/venv/bin:$CARGO_HOME/bin:$PATH

# Host compiler: GCC 14 (gcc-native/14, the PrgEnv default). Use gcc/g++ directly, not the Cray cc/CC wrappers.
export CC=gcc CXX=g++
GCC_DIR=$(dirname "$(g++ -print-libgcc-file-name)")      # /usr/lib64/gcc/x86_64-suse-linux/14
# ROCm's clang compiles the HIP sources and links the executable. Left alone it picks the newest GCC on the node
# (15 here); pin it to the host compiler's GCC so libstdc++ headers and libraries match.
HIPFLAGS="--rocm-path=$ROCM_PATH --gcc-install-dir=$GCC_DIR"

echo "node $(hostname)  $(date -Is)  jobs=$JOBS  arch=$ARCH"
echo "MMseqs2 commit $(git -C "$ROOT/src" rev-parse HEAD)  ($(git -C "$ROOT/src" log -1 --format=%cI))"
g++ --version | head -1
"$ROCM_PATH/lib/llvm/bin/clang++" --version | head -1
cmake --version | head -1; ninja --version; rustc --version; cargo --version
echo "GCC_DIR=$GCC_DIR"

t0=$(date +%s)
rm -rf "$ROOT/build"
cmake -S "$ROOT/src" -B "$ROOT/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$ROOT/install" \
    -DHAVE_AVX2=1 \
    -DENABLE_HIP=1 \
    -DCMAKE_HIP_ARCHITECTURES="$ARCH" \
    -DCMAKE_HIP_COMPILER="$ROCM_PATH/lib/llvm/bin/clang++" \
    -DCMAKE_HIP_COMPILER_ROCM_ROOT="$ROCM_PATH" \
    -DCMAKE_HIP_FLAGS="$HIPFLAGS" \
    -DCMAKE_PREFIX_PATH="$ROCM_PATH" \
    -DOpenMP_C_FLAGS=-fopenmp -DOpenMP_C_LIB_NAMES=gomp \
    -DOpenMP_CXX_FLAGS=-fopenmp -DOpenMP_CXX_LIB_NAMES=gomp \
    -DOpenMP_gomp_LIBRARY="$GCC_DIR/libgomp.so" \
    -DCMAKE_INSTALL_RPATH="$ROCM_PATH/lib" \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON
t1=$(date +%s); echo "[build] configure: $((t1 - t0)) s"

cmake --build "$ROOT/build" -j "$JOBS"
t2=$(date +%s); echo "[build] compile+link: $((t2 - t1)) s"
cmake --install "$ROOT/build" >/dev/null

B="$ROOT/install/bin/mmseqs"
echo "== binary: $B ($(du -h "$B" | cut -f1))"
"$B" version
echo "== dynamic libraries (outside the system dirs)"
ldd "$B" | grep -v -E "^\s*/lib64/|linux-vdso" | sed 's/ (0x[0-9a-f]*)//'
echo "== runpath"; readelf -d "$B" | grep -i -E "rpath|runpath" || true
echo "== embedded GPU code objects"; "$ROCM_PATH/bin/roc-obj-ls" "$B" 2>/dev/null | awk '{print $2}' | sort | uniq -c || true
echo "== GPU options"; "$B" search -h 2>&1 | grep -A1 -E "^\s+--gpu " || true
echo "[build] done $(date -Is)"
