#!/bin/bash
# bz-env.sh -- per-job environment for the Boltz-2 kit on Setonix MI250X (gfx90a).
#
#   source scripts/bz-env.sh
#
# Replaces configs/<card>.env (the kit's configs carry only MODEL_OPT, MODEL_OPT_TARGET_GPU, the JIT-cache key and
# PYTHONDONTWRITEBYTECODE; nothing in the kit tree is edited) plus what this MACHINE needs. A value already in the
# environment wins. Own variables use the B2S_ prefix (the kit strips BOLTZ_* / BZ_* / ROWPAIR_* families from its
# worker and refuses undeclared BOLTZ2_OPT* names).

D=${B2S_ROOT:-/scratch/pawsey1018/gbouras/boltz2-setonix}
AF3=${B2S_AF3_ROOT:-/scratch/pawsey1018/gbouras/af3-setonix}

# --- the base env must still be whole ------------------------------------------------------------------------------
# The 21-day /scratch purge deleted the base env's stdlib landmark (os.py) on 2026-09-24; every interpreter then dies at
# start in 0 s. Fail loudly here instead.
B2S_BASE_LANDMARK=${B2S_BASE_LANDMARK:-$AF3/py312/lib/python3.12/os.py}
if [ ! -e "$B2S_BASE_LANDMARK" ]; then
  echo "[b2s] BASE ENV PURGED: $B2S_BASE_LANDMARK is missing" >&2
  return 97 2>/dev/null || exit 97
fi

# --- ROCm 7.2.4 user space (the host default is 6.3.0; torch carries its own 7.2 runtime, the module gives the tools) --
module use /software/setonix/unsupported >/dev/null 2>&1 || true
module load rocm/7.2.4 >/dev/null 2>&1 || true

# --- the layered venv, and the nvidia-smi shim first on PATH ----------------------------------------------------------
# The kit probes the card with nvidia-smi only (boltz2_opt/stack.py gpu_probe / visible_gpus; opt_core/gates.py
# nvidia_smi_probe). Absent, every kit mode refuses ("no NVIDIA GPU (nvidia-smi found none)") and there is no override.
# scripts/bin/nvidia-smi answers those --query-gpu forms from sysfs (KFD topology): see its header for what it reports.
export VIRTUAL_ENV="$D/venv"
export PATH="$D/scripts/bin:$D/venv/bin:$PATH"

# --- what configs/<card>.env sets -------------------------------------------------------------------------------------
export MODEL_OPT="${MODEL_OPT:-$D/kit/boltz2}"
export MODEL_OPT_TARGET_GPU="${MODEL_OPT_TARGET_GPU:-MI250X}"     # a label: `check` notes the mismatch with the H100/A100 classes
# modes.jit_cache_key() reads torch.version.cuda, which is None on ROCm -> 'unknown'. Preset (it is only a directory component).
export MODEL_OPT_STACK_KEY="${MODEL_OPT_STACK_KEY:-torch2.14.0-rocm7.2-gfx90a}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

# --- node-local NVMe for every JIT / compile cache and temporary (never Lustre: inodes and metadata load) ---------------
: "${B2S_LOCAL:=/tmp/b2s-$USER-${SLURM_JOB_ID:-nojob}}"
mkdir -p "$B2S_LOCAL"/{tmp,xdg,jit,miopen,inductor,numba}
export B2S_LOCAL
export TMPDIR="$B2S_LOCAL/tmp" XDG_CACHE_HOME="$B2S_LOCAL/xdg"
export MODEL_OPT_JIT_ROOT="${MODEL_OPT_JIT_ROOT:-$B2S_LOCAL/jit}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$MODEL_OPT_JIT_ROOT/$MODEL_OPT_STACK_KEY/triton}"
export MIOPEN_USER_DB_PATH="$B2S_LOCAL/miopen" MIOPEN_CUSTOM_CACHE_DIR="$B2S_LOCAL/miopen"
export TORCHINDUCTOR_CACHE_DIR="$B2S_LOCAL/inductor" NUMBA_CACHE_DIR="$B2S_LOCAL/numba"
mkdir -p "$TRITON_CACHE_DIR"
# the weights' sha256 memo: persistent and tiny (else 4.7 GB is hashed again in every job)
export MODEL_OPT_WEIGHTS_DIGEST_DIR="${MODEL_OPT_WEIGHTS_DIGEST_DIR:-$D/cache/digest}"
mkdir -p "$MODEL_OPT_WEIGHTS_DIGEST_DIR"
# BOLTZ_CACHE is staged per job by scripts/stage-cache.sh (node-local: symlinks to $D/weights + mols/ unpacked).

# --- launch blocking: ON by default (2026-10-04) -------------------------------------------------------------------------
# Without it, stock Boltz-2 itself is unreliable from ~500-576 tokens on this stack: SIGSEGV in a different host frame every time
# (dropout mask, softmax, attention, Lightning's teardown) or a finite output and then a hang at exit (smoke 7, job 50365713).
# With it, every stock and kit run completes, and the outputs are byte-identical to runs that happened to finish without it
# (smoke 7 Z2 = Z8; smoke 8 W7 = Z2). exact's captured roll-out works under it (smoke 8 W2, W5: byte-identical to stock).
# Cost: 1.33x per item at 576 tokens (stock). HIP_LAUNCH_BLOCKING=0 restores asynchronous launches (small inputs only).
export HIP_LAUNCH_BLOCKING="${HIP_LAUNCH_BLOCKING:-1}"

# --- HIP graph packet capture: OFF by default (2026-10-09, the user's decision) -------------------------------------------
# ROCm 7.2 bug, found by the ProteinMPNN port: replaying a captured HIP graph corrupts the host heap (the fault is in ROCr's
# async-events thread). This switches CLR's packet capture off. With it fast's graphed sampler roll-out runs; without it the worker
# dies (SIGSEGV), launch blocking or not. It changed no byte on any path tested, and the kit does not strip it (graph fix, job 50560965).
export DEBUG_CLR_GRAPH_PACKET_CAPTURE="${DEBUG_CLR_GRAPH_PACKET_CAPTURE:-0}"

# --- allocator: deliberately NOT set ----------------------------------------------------------------------------------
# The kit's big row sets expandable_segments:True in its own worker (lever expandable_segments). That setting strands
# memory on ROCm 7.2 (Protenix ladder 50229219 / 50233809); it is a capacity issue, not a smoke-test one.

# --- threads: one Trento chiplet (8 cores) per GCD ---------------------------------------------------------------------
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

echo "[b2s] venv=$VIRTUAL_ENV kit=$MODEL_OPT stack_key=$MODEL_OPT_STACK_KEY local=$B2S_LOCAL HIP_LAUNCH_BLOCKING=${HIP_LAUNCH_BLOCKING:-unset} GRAPH_PACKET_CAPTURE=${DEBUG_CLR_GRAPH_PACKET_CAPTURE:-unset} ALLOC=${PYTORCH_CUDA_ALLOC_CONF:-unset}"
