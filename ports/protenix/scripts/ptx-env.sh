#!/bin/bash
# ptx-env.sh -- per-job environment for the Protenix v2 kit on Setonix MI250X (gfx90a).
#
#   source scripts/ptx-env.sh
#
# The kit's own launcher is used unchanged: `python -m protenix_opt pred --mode big
# --n_gpu P ...` (tp.py -> ptx_tp.launch -> python -m torch.distributed.run with a
# loopback c10d rendezvous; device = cuda:LOCAL_RANK). Its sharded-line environment
# (TP_PRE, TP_GATES, BIND_ENV, RANK_ENV_DEFAULTS, the PairCore defaults) is exported by
# the kit itself. What is set here is only what configs/h100.env would set, plus what
# this MACHINE needs. A value already in the environment wins.
#
# Own variables use the PXS_ prefix: the kit strips PTX_*/FPF_*/INFOPT_*/PF_* from the
# stock route and refuses undeclared PROTENIX_OPT* names at interpreter start.

D=${PXS_ROOT:-/scratch/pawsey1018/gbouras/protenix-setonix}

# --- the base env must still be whole ------------------------------------------------
# The 21-day /scratch purge deleted the base env's stdlib landmark (os.py) on 2026-09-24; every
# interpreter then dies at start in 0 s. Conda re-extraction stamps atime = mtime = the package date,
# so a repaired env can be swept again at the next pass. Fail loudly here instead.
PXS_BASE_LANDMARK=${PXS_BASE_LANDMARK:-/scratch/pawsey1018/gbouras/af3-setonix/py312/lib/python3.12/os.py}
if [ ! -e "$PXS_BASE_LANDMARK" ]; then
  echo "[pxs] BASE ENV PURGED: $PXS_BASE_LANDMARK is missing (see PROTENIX_PORT.md 'Environment incident')" >&2
  return 97 2>/dev/null || exit 97
fi

# --- ROCm 7.2.4 (host default is 6.3.0; the validated stack used 7.2.4) ------------
module use /software/setonix/unsupported >/dev/null 2>&1 || true
module load rocm/7.2.4 >/dev/null 2>&1 || true

# --- the layered venv ----------------------------------------------------------------
export VIRTUAL_ENV="$D/venv"
export PATH="$D/venv/bin:$PATH"

# --- what configs/<card>.env sets (the kit's configs differ only in the GPU label) ---
export MODEL_OPT="${MODEL_OPT:-$D/kit/protenix_v2}"
export MODEL_OPT_TARGET_GPU="${MODEL_OPT_TARGET_GPU:-MI250X}"          # a label; check/pred only note a mismatch
# opt_core.jit_cache.key() raises StackKeyUnknown on ROCm (torch.version.cuda is None,
# and there is no nvidia-smi for the cc): preset it. It is only a directory component.
export MODEL_OPT_STACK_KEY="${MODEL_OPT_STACK_KEY:-torch2.14.0-rocm7.2-gfx90a}"
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-$D/weights}"            # checkpoint/protenix-v2.pt + common/
# PROTENIX_ROOT_FROZEN is deliberately NOT set: the kit's start-up hook then checks the
# frozen inputs in EVERY interpreter against the stock CLI's default model name
# (protenix_base_default_v1.0.0), so even a plain `python -c 'import torch'` exits 3
# (probe 49904583). All seven pinned files are present (WEIGHTS OK 7/7), so nothing downloads.
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"                            # the kit hands the ranks 0 anyway

# --- no nvidia-smi on Setonix -----------------------------------------------------------
# stack._gates probes the GPU with nvidia-smi only; absent, every kit mode refuses with
# "no CUDA device visible" unless PROTENIX_OPT_FORCE=1 (documented: it turns the version
# and no-CUDA gates into warnings, nothing else). The consequence is deliberate: with no
# nvidia-smi compute capability the kernel key is unknown, so modes.resolve takes the
# README's "other" row (env.sh's generic ARM defaults) rather than the H100 row that
# torch's (9,0) would select. The opt_core providers still read (9,0) from torch.
export PROTENIX_OPT_FORCE="${PROTENIX_OPT_FORCE:-1}"

# --- stock LayerNorm: torch ---------------------------------------------------------
# stock's default (fast_layernorm) JIT-builds an apex-derived CUDA extension with
# nvcc -gencode flags at import; there is no nvcc here, and the kit's prebuilt .so files
# are torch 2.7.1+cu126 / 2.13.0+cu130 builds.
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch}"
# ...which makes two LayerNorm levers unavailable, and a mode refuses a partial lever set
# (exit 3) unless they are ablated by name (probe 49904583):
#   fastln_prebuilt  no third_party/fastln_prebuilt*/manifest.json matches torch 2.14.0+rocm7.2
#   ln_core          binds stock's fast_layernorm module, which LAYERNORM_TYPE=torch removes
#   sampler_fuse     the DiT's fused elementwise kernels call libdevice.div_rn, which
#                    triton.language.extra.hip.libdevice does not have: CompilationError in
#                    _res_gate_kernel kills the item at the first sampler step (probe 49904674)
# Further kernel-specific ablations are appended by the job scripts, each with its reason.
export MODEL_OPT_LEVERS_OFF="${MODEL_OPT_LEVERS_OFF:-fastln_prebuilt,ln_core,sampler_fuse}"

# --- node-local NVMe for every JIT / compile cache (never Lustre) -------------------
: "${PXS_LOCAL:=/tmp/ptx-$USER-${SLURM_JOB_ID:-nojob}}"
mkdir -p "$PXS_LOCAL"/{tmp,xdg,jit,miopen,inductor,numba,records}
export PXS_LOCAL
export TMPDIR="$PXS_LOCAL/tmp"        # also where the TP line puts its per-run rank file store
export XDG_CACHE_HOME="$PXS_LOCAL/xdg"
export MODEL_OPT_JIT_ROOT="${MODEL_OPT_JIT_ROOT:-$PXS_LOCAL/jit}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$MODEL_OPT_JIT_ROOT/$MODEL_OPT_STACK_KEY/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$MODEL_OPT_JIT_ROOT/$MODEL_OPT_STACK_KEY/torch_extensions}"
export PROTENIX_OPT_CACHE_DIR="${PROTENIX_OPT_CACHE_DIR:-$D/cache/protenix_opt}"   # the weights digest memo: persistent, tiny
export MIOPEN_USER_DB_PATH="$PXS_LOCAL/miopen" MIOPEN_CUSTOM_CACHE_DIR="$PXS_LOCAL/miopen"
export TORCHINDUCTOR_CACHE_DIR="$PXS_LOCAL/inductor" NUMBA_CACHE_DIR="$PXS_LOCAL/numba"
mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$PROTENIX_OPT_CACHE_DIR"

# --- rocBLAS gfx90a Tensile gap -------------------------------------------------------
# =1 works around it above ~1,000 residues; =0 is ~1.17x faster where it fits but raises
# transient memory (OpenFold3 OOM'd at 13,090 with =0 where =1 fits). Ceiling probes: 1.
export HIP_LAUNCH_BLOCKING="${HIP_LAUNCH_BLOCKING:-1}"

# --- the sharded line's DiT attention: the engine statement --------------------------
# --n_gpu P only (read by protenix_opt.tp_bind.diffusion; stock and the one-GCD line never
# read it). The kit's default rows word 'big' has no measured rows cell on gfx90a, so it
# falls back by name to apb_attn with k / v cast to bf16: 2,400-3,000x stock's fp32
# denoiser error (job 50058450). 'engine' runs the engine's own attention statement per
# query block: 85-110x at high noise beside the default fp16 bias cache, at no measurable
# memory or time cost from 2,048 to 12,288 tokens (the DiT 6 % faster at 12,288; job
# 50072061). ROWPAIR_DIT_ROWS=big restores the kit's default, which every run before
# 2026-09-29 used unless it set this word. For stock's precision exactly, add
# ROWPAIR_DIFF_BIAS_CACHE_DTYPE=fp32: free while the cache fits (to ~12,000 tokens on
# 8 GCDs), then 8.7x per sampler step, because the kit recomputes the biases instead.
export ROWPAIR_DIT_ROWS="${ROWPAIR_DIT_ROWS:-engine}"

# --- the sharded line's triangle attention: opt_core's fused flash_triattn ---------------
# --n_gpu P: ROWPAIR_TRIATT_CORE=flash_triattn serves triangle attention through the kit's
# own dispatch (opt_core.mem.rowpair.triatt.attention_core: launches split under int32,
# larger planes one query block at a time, the bias staged once per plane) in place of
# stock's torch statement. OF3T_FLASH_IP=ieee: the kernel's default dot precision (tf32) is
# not an AMD precision and does not build on gfx90a; it is read when the module is
# imported, so it must be set before Python starts. PTX_TRIATTN_CONFIG: the OpenFold3
# port's gfx90a tiles for D=32 (the kernel's table otherwise gives H100's). Within stock's
# error class per op and per whole block (0.98-1.00x against fp64; jobs 50350451, 50379609).
# NCCL_NCHANNELS_PER_PEER=8 (RCCL point-to-point channels per peer, default 2) and
# TORCH_NCCL_HIGH_PRIORITY=1: bitwise identical results (job 50379609). Together: 3.1x on a
# 6,144-token run on 8 GCDs, the trunk 3.9x (job 50350451). The one-GCD line is unchanged:
# with --triatt_kernel torch its block never reaches the kit's own fused attention (K2B), and
# a deterministic 1BRS run with all five set is identical to stock (job 50379609).
# ROWPAIR_TRIATT_CORE=torch restores the eager statement, which every run before 2026-10-05
# used.
export ROWPAIR_TRIATT_CORE="${ROWPAIR_TRIATT_CORE:-flash_triattn}"
export OF3T_FLASH_IP="${OF3T_FLASH_IP:-ieee}"
export PTX_TRIATTN_CONFIG="${PTX_TRIATTN_CONFIG:-128,32,2,4,1,1}"
export NCCL_NCHANNELS_PER_PEER="${NCCL_NCHANNELS_PER_PEER:-8}"
export TORCH_NCCL_HIGH_PRIORITY="${TORCH_NCCL_HIGH_PRIORITY:-1}"

# --- allocator ------------------------------------------------------------------------
# Not set here: the kit's XL policy writes PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# (env.sh) and the TP launcher hands its own pre-CUDA exports to the ranks.

# --- threads ----------------------------------------------------------------------------
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

echo "[pxs] venv=$VIRTUAL_ENV kit=$MODEL_OPT weights=$PROTENIX_ROOT_DIR"
echo "[pxs] local=$PXS_LOCAL stack_key=$MODEL_OPT_STACK_KEY LAYERNORM_TYPE=$LAYERNORM_TYPE FORCE=$PROTENIX_OPT_FORCE HIP_LAUNCH_BLOCKING=$HIP_LAUNCH_BLOCKING DIT_ROWS=$ROWPAIR_DIT_ROWS TRIATT_CORE=$ROWPAIR_TRIATT_CORE LEVERS_OFF=$MODEL_OPT_LEVERS_OFF"
