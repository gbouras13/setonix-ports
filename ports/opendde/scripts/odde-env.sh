#!/bin/bash
# odde-env.sh -- per-job environment for the OpenDDE kit on Setonix MI250X (gfx90a), inside the image.
#
#   source scripts/odde-env.sh
#
# The native port's scripts/odde-env.sh (af3-setonix/opendde/), with the image's paths: the tree is /opt/opendde (ODDE_ROOT), its
# interpreter is the image's own, and torch carries its own ROCm 7.2 runtime, so no module is loaded. Everything that sets numerics
# is the native file's, value for value. OpenDDE's CLI launches its own rank processes (cli.pred_parent -> tp.run_ranks ->
# opt_core.mem.rowpair.launch.run_rank_processes) and exports the sharded line's variables itself; what is set here is only what the
# MACHINE needs. A value already in the environment wins.

D=${ODDE_ROOT:-/opt/opendde}
export ODDE_ROOT=$D                                   # the patch hooks find each other (and the per-rank hook) through it

# --- the tree must be whole ---------------------------------------------------------------------------------------------
if [ ! -x "$D/venv/bin/python" ] || [ ! -d "$D/kit/opendde" ]; then
  echo "[odde] INCOMPLETE TREE: $D/venv/bin/python or $D/kit/opendde is missing" >&2
  return 97 2>/dev/null || exit 97
fi

# --- the venv -----------------------------------------------------------------------------------------------------------
export VIRTUAL_ENV="$D/venv"
export PATH="$D/venv/bin:$PATH"

# --- the kit ------------------------------------------------------------------------------------------------------------
export MODEL_OPT="${MODEL_OPT:-$D/kit/opendde}"
export OPENDDE_ROOT_DIR="${OPENDDE_ROOT_DIR:-${ODDE_WEIGHTS:-$D/weights}}"   # the image has no weights: give ODDE_WEIGHTS
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"        # the kit's own images set it

# The per-rank cache hook (scripts/hooks/sitecustomize.py): 8 ranks sharing one Triton cache directory is a known corruption source.
export PYTHONPATH="$D/scripts/hooks${PYTHONPATH:+:$PYTHONPATH}"

# --- node-local disk for every JIT / compile cache and temporary ---------------------------------------------------------
# /tmp on a Setonix GPU node is NVMe (128 GB). Lustre is fatal here: Triton/MIOpen stat thousands of small files per launch.
: "${ODDE_LOCAL:=${TMPDIR:-/tmp}/odde-$(id -u)-${SLURM_JOB_ID:-nojob}}"
mkdir -p "$ODDE_LOCAL"/{tmp,xdg,jit,miopen,inductor,numba}
export ODDE_LOCAL
export TMPDIR="$ODDE_LOCAL/tmp"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ODDE_LOCAL/xdg}"
export MODEL_OPT_JIT_ROOT="${MODEL_OPT_JIT_ROOT:-$ODDE_LOCAL/jit}"
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$ODDE_LOCAL/miopen}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CUSTOM_CACHE_DIR:-$ODDE_LOCAL/miopen}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ODDE_LOCAL/inductor}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$ODDE_LOCAL/numba}"

# The kit keys its JIT caches by a stack key (modes.jit_cache_key, a directory component only). Preset: computing it starts an interpreter.
export MODEL_OPT_STACK_KEY="${MODEL_OPT_STACK_KEY:-torch2.14.0-rocm7.2-gfx90a}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$MODEL_OPT_JIT_ROOT/$MODEL_OPT_STACK_KEY/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$MODEL_OPT_JIT_ROOT/$MODEL_OPT_STACK_KEY/torch_ext}"
# The weights' sha256 memo: the entry point puts it beside the weights when that directory is writable (2.6 GB hashed once).
export MODEL_OPT_WEIGHTS_DIGEST_DIR="${MODEL_OPT_WEIGHTS_DIGEST_DIR:-$MODEL_OPT_JIT_ROOT/weights}"

# --- what gfx90a cannot do ----------------------------------------------------------------------------------------------
# lnstream JIT-compiles upstream's apex-derived CUDA LayerNorm with nvcc + ninja; there is no nvcc. Upstream's own default is torch.
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"

# big's two size gates. card_memory_mib() reads nvidia-smi only, so on ROCm the probe returns None and gate_default falls through to
# the 80 GB row; and even a working probe would take it, because SMALL_CARD_MIB is 64*1024 and 65536 is not < 65536. Use the
# sub-64-GiB values the kit itself carries (big.py SMALL_CARD_GATES).
export MODEL_OPT_BIG_OFFLOAD_MIN_TOKENS="${MODEL_OPT_BIG_OFFLOAD_MIN_TOKENS:-1160}"
export MODEL_OPT_BIG_NO_DIT_HOIST_MIN_TOKENS="${MODEL_OPT_BIG_NO_DIT_HOIST_MIN_TOKENS:-1856}"
export MODEL_OPT_SMALL_INPUT_FLOOR_TOKENS="${MODEL_OPT_SMALL_INPUT_FLOOR_TOKENS:-300}"
export MODEL_OPT_TARGET_GPU="${MODEL_OPT_TARGET_GPU:-MI250X}"   # a label only; rows are chosen by the running device

# Launch blocking: REQUIRED on the sharded line. With HIP_LAUNCH_BLOCKING=0 rank 7 segfaulted in the diffusion roll-out in 2 of 2
# runs at 4,096 tokens (af3-setonix/OPENDDE_PORT_LOG.md, job 50309590).
export HIP_LAUNCH_BLOCKING="${HIP_LAUNCH_BLOCKING:-1}"

# --- allocator: deliberately NOT set here --------------------------------------------------------------------------------
# The kit writes PYTORCH_CUDA_ALLOC_CONF itself (expandable_segments:True on the big line); the patched line's native-allocator
# patch turns that into :False through the kit's own writer (patches/odde_native_alloc_patch.py).

# --- threads: one Trento chiplet (8 cores) per GCD; the sharded line divides the cpuset itself (ROWPAIR_RANK_THREADS=auto) ----
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

echo "[odde] tree $D  kit $MODEL_OPT  weights $OPENDDE_ROOT_DIR  local $ODDE_LOCAL  HIP_LAUNCH_BLOCKING=$HIP_LAUNCH_BLOCKING" \
     "gates offload>=$MODEL_OPT_BIG_OFFLOAD_MIN_TOKENS hoist_off>=$MODEL_OPT_BIG_NO_DIT_HOIST_MIN_TOKENS"
