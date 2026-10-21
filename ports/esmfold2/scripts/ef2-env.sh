#!/bin/bash
# ef2-env.sh -- per-job environment for the ESMFold2 kit on Setonix MI250X (gfx90a, ROCm 7.2).
#
#   source scripts/ef2-env.sh
#
# What it does NOT replace: configs/h100.env. Every value that file sets is `${VALUE:-default}`, so this file pre-sets the
# machine's words and the kit's config fills the rest and runs its own probes (scripts/ef2-run.sh sources it, as run.sh does).
# Own variables use the EFS_ prefix: the kit's stock subprocess must not see EF2_*, ESMFOLD2_OPT*, ESMFOLD2_VARIANT,
# CUBLAS_WORKSPACE_CONFIG, NVIDIA_TF32_OVERRIDE, FPF_TRIMUL_V4_CELLS or CUDA_MPS_* (stock/PINS.json "stock_environment"; its
# env_proof refuses them), so nothing of ours may use those names.
#
# The venv's base interpreter is cray-python 3.12.12 (/opt/cray/pe/python, a system path the /scratch purge cannot touch);
# the venv itself lives on /scratch and is rebuilt by scripts/setup.sh (REBUILD.md). In the setonix-ports image the tree is
# /opt/esmfold2 (EFS_ROOT), read-only: its entry point presets MODEL_OPT_JIT_ROOT, TORCH_HOME and HF_HOME to writable places.

D=${EFS_ROOT:-/scratch/pawsey1018/gbouras/esmfold2-setonix}
if [ ! -x "$D/venv/bin/python" ] || [ ! -e "$D/venv/pyvenv.cfg" ]; then
  echo "[efs] NO VENV at $D/venv (rebuild: REBUILD.md)" >&2
  return 97 2>/dev/null || exit 97
fi

# --- ROCm 7.2.4 user space (tools; torch carries its own 7.2 runtime). The host default is 6.3.0. ----------------------------
module use /software/setonix/unsupported >/dev/null 2>&1 || true
module load rocm/7.2.4 >/dev/null 2>&1 || true

# --- the venv, and the nvidia-smi shim first on PATH ---------------------------------------------------------------------
# opt_core.gates identifies the GPU with nvidia-smi for the dry run (esmfold2_opt/stack.py nvidia_smi_probe); scripts/bin/nvidia-smi
# answers those queries from the KFD topology in sysfs (name, memory, compute_cap 9.0 = what torch reports on gfx90a).
export VIRTUAL_ENV="$D/venv"
export PATH="$D/scripts/bin:$VIRTUAL_ENV/bin:$PATH"

# --- what configs/h100.env would otherwise fill ---------------------------------------------------------------------------
export MODEL_OPT="${MODEL_OPT:-$D/kit/esmfold2}"
export MODEL_OPT_TARGET_GPU="${MODEL_OPT_TARGET_GPU:-MI250X}"        # a label; the kit prints a NOTE for a card it does not target
# the JIT cache key is derived from torch.version.cuda, unset on ROCm: preset it (the earlier ports' key format)
export MODEL_OPT_STACK_KEY="${MODEL_OPT_STACK_KEY:-torch2.14.0-rocm7.2-gfx90a}"
export MODEL_OPT_JIT_ROOT="${MODEL_OPT_JIT_ROOT:-$D/cache/jit}"      # persistent Triton / extension caches (never $HOME: 1 GB quota)
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$MODEL_OPT_JIT_ROOT/$MODEL_OPT_STACK_KEY/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$MODEL_OPT_JIT_ROOT/$MODEL_OPT_STACK_KEY/torch_extensions}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$MODEL_OPT_JIT_ROOT/$MODEL_OPT_STACK_KEY/inductor}"
export ESMFOLD2_OPT_WEIGHTS_MEMO_DIR="${ESMFOLD2_OPT_WEIGHTS_MEMO_DIR:-$MODEL_OPT_JIT_ROOT/weights}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$ESMFOLD2_OPT_WEIGHTS_MEMO_DIR" 2>/dev/null || true
export HF_HOME="${HF_HOME:-$D/weights}"                             # the weights root: the three pinned snapshots (scripts/setup.sh step 5)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1                      # nothing is fetched at run time (configs/h100.env sets the same)
# The pinned image's CUDA accelerated layer (flash-attn, transformer-engine, xformers) does not exist for this machine: upstream then
# runs its own fallbacks (atom_attn=sdpa esmc_mlp=torch esmc_rope=torch). The kit's documented setting for such a stack is 0 (report
# the words on every route and proceed); its configs default to 1 (refuse). Override with EFS_REQUIRE_FAST_ENV=1 to see the refusal.
export ESMFOLD2_OPT_REQUIRE_FAST_ENV="${EFS_REQUIRE_FAST_ENV:-0}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}" NVTE_FRAMEWORK=pytorch XFORMERS_IGNORE_FLASH_VERSION_CHECK=1   # the kit image's ENV (STOCK.md)

# --- node-local NVMe for temporaries ---------------------------------------------------------------------------------------
: "${EFS_LOCAL:=/tmp/efs-$USER-${SLURM_JOB_ID:-nojob}}"
mkdir -p "$EFS_LOCAL"/{tmp,xdg,miopen}
export EFS_LOCAL TMPDIR="$EFS_LOCAL/tmp" XDG_CACHE_HOME="$EFS_LOCAL/xdg"
export MIOPEN_USER_DB_PATH="$EFS_LOCAL/miopen" MIOPEN_CUSTOM_CACHE_DIR="$EFS_LOCAL/miopen"
export TORCH_HOME="${TORCH_HOME:-$D/cache/torch}" PIP_CACHE_DIR="$EFS_LOCAL/pip"

# --- HIP graph packet capture (ROCm 7.2) -----------------------------------------------------------------------------------
# Repeated HIP graph replay corrupts the host heap / SIGSEGVs on ROCm 7.2 unless DEBUG_CLR_GRAPH_PACKET_CAPTURE=0 (confirmed in
# ProteinMPNN's decode loop and Boltz-2's sampler; default in the RFdiffusion3 port). The kit's exact / fast capture the trunk,
# encoders and the sampler, so this port starts with capture off; EFS_GRAPH_PACKET_CAPTURE=1 puts ROCm's default back for the A/B.
export DEBUG_CLR_GRAPH_PACKET_CAPTURE="${EFS_GRAPH_PACKET_CAPTURE:-${DEBUG_CLR_GRAPH_PACKET_CAPTURE:-0}}"
# launch blocking: off unless asked (AF3 JAX / OpenDDE needed it; determinism repeats decide it here)
[ -z "${EFS_LAUNCH_BLOCKING:-}" ] || export HIP_LAUNCH_BLOCKING="$EFS_LAUNCH_BLOCKING"

# --- allocator: left as the kit sets it unless EFS_ALLOC says otherwise ---------------------------------------------------
# `big` requires expandable_segments:True and refuses another value; exact / fast export it too but keep a value the caller
# set (stack.graph_lines_alloc_export); stock (--mode off) runs torch's default.
case "${EFS_ALLOC:-}" in
  True|true)   export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" ;;
  False|false) export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False" ;;
esac

# --- start-up fixes applied in memory, never in the kit tree ------------------------------------------------------------
# EFS_BOOT names them (scripts/boot/sitecustomize.py), per RUN; unset = nothing is armed. The directory goes on PYTHONPATH
# unconditionally (a job sources this file once, EFS_BOOT is chosen per run). The stock route is never run with EFS_BOOT set.
case ":${PYTHONPATH:-}:" in
  *":$D/scripts/boot:"*) ;;
  *) export PYTHONPATH="$D/scripts/boot${PYTHONPATH:+:$PYTHONPATH}" ;;
esac

# --- threads: one Trento chiplet (8 cores) per GCD --------------------------------------------------------------------------
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

echo "[efs] venv=$VIRTUAL_ENV kit=$MODEL_OPT stack_key=$MODEL_OPT_STACK_KEY HF_HOME=$HF_HOME local=$EFS_LOCAL" \
     "REQUIRE_FAST_ENV=$ESMFOLD2_OPT_REQUIRE_FAST_ENV GRAPH_PACKET_CAPTURE=$DEBUG_CLR_GRAPH_PACKET_CAPTURE" \
     "LAUNCH_BLOCKING=${HIP_LAUNCH_BLOCKING:-unset} ALLOC=${PYTORCH_CUDA_ALLOC_CONF:-unset(kit default)}"
