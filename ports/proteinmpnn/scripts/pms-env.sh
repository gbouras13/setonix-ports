#!/bin/bash
# pms-env.sh -- per-job environment for the ProteinMPNN kit on Setonix MI250X (gfx90a).
#
#   source scripts/pms-env.sh
#
# Replaces configs/<card>.env (MODEL_OPT, MODEL_OPT_TARGET_GPU[_MEM_MIB], PYTHONDONTWRITEBYTECODE; nothing in the kit tree is
# edited) plus what this MACHINE needs. A value already in the environment wins. Own variables use the PMS_ prefix: the kit
# strips PROTEINMPNN_* / MPNN_* (except MPNN_DIR) / SELFTEST_* / PYTORCH_CUDA_ALLOC_CONF / CUBLAS_WORKSPACE_CONFIG from the
# stock subprocess (stock/PINS.json must_be_absent_prefixes), so nothing of ours may use those prefixes.
#
# The venv's base interpreter is cray-python 3.11.7 (/opt/cray/pe/python, a system path the /scratch purge cannot touch);
# the venv itself lives on /scratch and is rebuilt by scripts/setup.sh (REBUILD.md).

D=${PMS_ROOT:-/scratch/pawsey1018/gbouras/proteinmpnn-setonix}

# --- the venv and its base must be whole -------------------------------------------------------------------------------
if [ ! -x "$D/venv/bin/python" ] || [ ! -e "$D/venv/pyvenv.cfg" ]; then
  echo "[pms] NO VENV at $D/venv (rebuild: REBUILD.md)" >&2
  return 97 2>/dev/null || exit 97
fi

# --- ROCm 7.2.4 user space (tools only: torch carries its own 7.2 runtime; the host default is 6.3.0) --------------------
module use /software/setonix/unsupported >/dev/null 2>&1 || true
module load rocm/7.2.4 >/dev/null 2>&1 || true

# --- the venv, and the nvidia-smi shim first on PATH ----------------------------------------------------------------------
# The kit identifies the GPU with nvidia-smi only (proteinmpnn_opt/stack.py gpu_info). Absent, gpu=None and the exact line
# is refused at activation (its CUDA-graph levers cannot run "without a CUDA device"). scripts/bin/nvidia-smi answers from sysfs.
export VIRTUAL_ENV="$D/venv"
export PATH="$D/scripts/bin:$D/venv/bin:$PATH"

# --- what configs/<card>.env sets ---------------------------------------------------------------------------------------
export MODEL_OPT="${MODEL_OPT:-$D/kit/proteinmpnn}"
export MPNN_DIR="${MPNN_DIR:-$D/ProteinMPNN}"                     # upstream's clone at 8907e667 (code, .git, both weight sets)
export MODEL_OPT_TARGET_GPU="${MODEL_OPT_TARGET_GPU:-MI250X}"     # a label: the activation line reports match= against it
export MODEL_OPT_TARGET_GPU_MEM_MIB="${MODEL_OPT_TARGET_GPU_MEM_MIB:-65520}"   # one MI250X GCD as the shim reports it
# stack_key / jit_cache.key read torch.version.cuda (None on ROCm): preset; it is only a directory component of the JIT cache.
export MODEL_OPT_STACK_KEY="${MODEL_OPT_STACK_KEY:-torch2.14.0-rocm7.2-gfx90a}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"                      # the kit images' process environment (STOCK.md)

# --- node-local NVMe for temporaries; the one Triton kernel's cache persistent on /scratch (tiny) -------------------------
: "${PMS_LOCAL:=/tmp/pms-$USER-${SLURM_JOB_ID:-nojob}}"
mkdir -p "$PMS_LOCAL"/{tmp,xdg,miopen}
export PMS_LOCAL TMPDIR="$PMS_LOCAL/tmp" XDG_CACHE_HOME="$PMS_LOCAL/xdg"
export MIOPEN_USER_DB_PATH="$PMS_LOCAL/miopen" MIOPEN_CUSTOM_CACHE_DIR="$PMS_LOCAL/miopen"
# The kit's exact line compiles one Triton kernel (the fused draw; ours under PMS_DRAW=rocm). run.sh keys TRITON_CACHE_DIR
# under MODEL_OPT_JIT_ROOT; a persistent root avoids a recompile per job (a few files).
export MODEL_OPT_JIT_ROOT="${MODEL_OPT_JIT_ROOT:-$D/cache/jit}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$MODEL_OPT_JIT_ROOT/$MODEL_OPT_STACK_KEY/triton}"
mkdir -p "$TRITON_CACHE_DIR" 2>/dev/null || true

# --- HIP graph packet capture OFF (2026-10-08) ---------------------------------------------------------------------------
# The kit's exact line replays one CUDA graph per decode step, hundreds of times back to back. With ROCm 7.2's default graph packet
# capture (CLR flag DEBUG_CLR_GRAPH_PACKET_CAPTURE=1) the process dies in that replay loop on longer inputs: SIGSEGV in ROCr's
# async-events thread (rocr::core::Runtime::AsyncEventsLoop, gdb, smoke 3d 50511887) and glibc heap corruption ("malloc(): unsorted
# double linked list corrupted", smoke 1-2: 3HTN, 4YOW, 6EHB, bench92). It is not about missing residues (5L33 with a gap passes,
# 3HTN without one crashes) nor launch blocking, lanes, streams or batch width (smoke 2 Z2-Z8). With capture off the same runs complete
# and are byte-identical to stock (smoke 3d G1 3HTN, P1 both complexes). Stock uses no graphs: no effect there.
export DEBUG_CLR_GRAPH_PACKET_CAPTURE="${DEBUG_CLR_GRAPH_PACKET_CAPTURE:-0}"

# --- launch blocking: NOT set (stock and exact are byte-identical with and without it: smoke 1 S6) -------------------------
# --- allocator: NOT set (expandable_segments:True strands memory on ROCm 7.2; the kit strips PYTORCH_CUDA_ALLOC_CONF from stock)

# --- threads: one Trento chiplet (8 cores) per GCD -----------------------------------------------------------------------
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

echo "[pms] venv=$VIRTUAL_ENV kit=$MODEL_OPT MPNN_DIR=$MPNN_DIR stack_key=$MODEL_OPT_STACK_KEY local=$PMS_LOCAL HIP_LAUNCH_BLOCKING=${HIP_LAUNCH_BLOCKING:-unset} ALLOC=${PYTORCH_CUDA_ALLOC_CONF:-unset} GRAPH_PACKET_CAPTURE=${DEBUG_CLR_GRAPH_PACKET_CAPTURE}"
