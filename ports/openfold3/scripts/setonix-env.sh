#!/bin/bash
# setonix-env.sh — environment for OpenFold3 inference on Setonix MI250X GCDs.
#
#   source scripts/setonix-env.sh
#
# Sets three things: a private node-local scratch for every JIT/compile cache,
# the HIP allocator policy, and thread counts. Everything is overridable -- a
# value already set in the environment wins.
#
# ---------------------------------------------------------------------------
# WHY THE CACHE PATHS MATTER MORE THAN ANYTHING ELSE HERE
# ---------------------------------------------------------------------------
# Triton, TorchInductor and MIOpen each write thousands of small files and stat
# them on every kernel launch path. Their defaults live under $HOME, which on
# Setonix is Lustre. Pointing them at Lustre gives you two separate problems:
#
#   1. Your job crawls. Lustre metadata latency on thousands of small-file stats
#      is orders of magnitude worse than a local filesystem.
#   2. You degrade the filesystem for everyone else on the machine. Pawsey will,
#      reasonably, come and find you.
#
# So: caches go to node-local scratch, seeded once from a pre-warmed archive
# (scripts/stage-caches.sh) rather than recompiled per job.
# ---------------------------------------------------------------------------

# --- node-local scratch -----------------------------------------------------
# Setonix GPU nodes have REAL local NVMe, not a RAM disk. Measured on nid002920
# (gpu-dev, 2026-09-18):
#     /tmp on /dev/nvme0n1, 128 GB, empty
#     TMPDIR=/tmp, SLURM_TMPDIR unset
# The partition advertises `tmp:350` as a Slurm GRES; what a job actually sees on
# a node was 128 GB. Either way it is local NVMe and it is the right home for the
# JIT caches. It does not survive the job, which is why stage-caches.sh seeds it
# from a persistent warmed archive rather than recompiling every time.
: "${AF3_SETONIX_LOCAL:=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/af3-setonix-$USER-${SLURM_JOB_ID:-nojob}}"
mkdir -p "$AF3_SETONIX_LOCAL"/{triton,inductor,miopen,numba}
export AF3_SETONIX_LOCAL

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$AF3_SETONIX_LOCAL/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$AF3_SETONIX_LOCAL/inductor}"
# MIOpen keeps a perf database AND a compiled-kernel cache. Both must move.
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$AF3_SETONIX_LOCAL/miopen}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CUSTOM_CACHE_DIR:-$AF3_SETONIX_LOCAL/miopen}"
# Numba is pulled in by parts of the featurisation stack and will try to write
# next to its source files if this is unset -- which fails on a read-only
# container filesystem. Pawsey hit this with Boltz too.
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$AF3_SETONIX_LOCAL/numba}"

# --- persistent (Lustre) state ----------------------------------------------
# Weights are large and read-once-per-job; they belong on Lustre, not /tmp.
# $MYSOFTWARE is the Pawsey-provided per-user persistent area.
# NOTE: $MYSOFTWARE points at your DEFAULT project, which may not be the one
# holding the GPU allocation -- on this account it resolves to director2187 while
# the work lives under pawsey1018, and OpenFold3 then dies with
#   PermissionError: .../director2187/gbouras/openfold3/runner.yml
# Prefer the kit's own workdir, which is where stage-weights.sh puts them.
if [ -z "${OPENFOLD_CACHE:-}" ]; then
    if [ -n "${AF3_SETONIX_WORKDIR:-}" ] && [ -d "$AF3_SETONIX_WORKDIR/openfold3-cache" ]; then
        export OPENFOLD_CACHE="$AF3_SETONIX_WORKDIR/openfold3-cache"
    else
        export OPENFOLD_CACHE="${MYSOFTWARE:-$HOME}/openfold3"
    fi
fi

# --- HIP allocator ----------------------------------------------------------
# expandable_segments lets the caching allocator grow a segment instead of
# fragmenting into many fixed blocks. On a 64 GB GCD running a large complex,
# fragmentation is frequently what actually kills the job -- the failure reads
# as OOM with plenty of "free" memory reported.
#
# torch on ROCm reads PYTORCH_HIP_ALLOC_CONF; older builds still read the CUDA
# spelling. Setting both is harmless and covers the ROCm 6.2 -> 6.3 span of
# containers on Setonix.
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- threads ----------------------------------------------------------------
# A Setonix GPU node is one 64-core Trento CPU across 8 chiplets, one chiplet per
# GCD. Measured with --gres=gpu:1 on nid002920: the job's cpuset is cores 0-7
# (SLURM_CPUS_ON_NODE=16, i.e. 8 physical cores with SMT2), and
# ROCR_VISIBLE_DEVICES=0 with HIP_VISIBLE_DEVICES unset -- exactly one GCD,
# 64 GiB VRAM. So 8 is the right default; letting OpenMP spawn 64 threads into an
# 8-core cpuset causes contention, not speed.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

# --- device visibility: a warning, not a setting ----------------------------
# Slurm sets ROCR_VISIBLE_DEVICES from --gres=gpu:N. Do NOT also set
# HIP_VISIBLE_DEVICES: the two filters compose, so ROCR exposing GCD 3 as index 0
# and HIP then selecting "device 3" addresses a GCD you were not allocated, or
# nothing at all. Leave both alone and let Slurm own it.
if [[ -n "${HIP_VISIBLE_DEVICES:-}" && -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    echo "[af3-setonix] WARNING: both HIP_VISIBLE_DEVICES (${HIP_VISIBLE_DEVICES}) and" >&2
    echo "[af3-setonix]          ROCR_VISIBLE_DEVICES (${ROCR_VISIBLE_DEVICES}) are set." >&2
    echo "[af3-setonix]          These compose. Unset HIP_VISIBLE_DEVICES." >&2
fi

echo "[af3-setonix] cache root : $AF3_SETONIX_LOCAL"
echo "[af3-setonix] weights    : $OPENFOLD_CACHE"
echo "[af3-setonix] alloc conf : $PYTORCH_HIP_ALLOC_CONF"
echo "[af3-setonix] threads    : $OMP_NUM_THREADS"
