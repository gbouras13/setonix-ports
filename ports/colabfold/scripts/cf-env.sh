#!/bin/bash
# cf-env.sh -- the process environment of the ColabFold port on Setonix GPU nodes (MI250X / gfx90a, ROCm 7.2.4). Source it inside a job
# (never on a login node). Every value can be pre-set by the caller; this file fills what is unset. Settings carried over from the af3_jax
# port (AF3JAX_PORT.md) unless noted:
#   - jax 0.10.2 + jax-rocm7 0.10.2 load the rocm/7.2.4 module's HIP runtime (module use /software/setonix/unsupported);
#   - XLA_PYTHON_CLIENT_MEM_FRACTION only: with XLA_CLIENT_MEM_FRACTION ALSO set, jax 0.10.2 raises inside the ROCm plugin's initialize() and
#     silently runs on the CPU; colabfold/batch.py sets XLA_PYTHON_CLIENT_MEM_FRACTION=4.0 (and TF_FORCE_UNIFIED_MEMORY=1) when they are
#     unset, so the port always sets both (the kit's own stock exception: STOCK.md 'Stock exceptions');
#   - JAX_PLATFORMS=rocm,cpu: a ROCm plugin that does not initialise is an ERROR, never a silent CPU run (jobs also assert the backend);
#   - JAX_PALLAS_USE_MOSAIC_GPU=0: Pallas lowers through Triton (Mosaic GPU refuses ROCm);
#   - XLA_FLAGS: command buffers off (HIP-graph failures), XLA's Triton softmax rewriter off (its 128 KiB tile exceeds gfx90a's 64 KiB LDS).
#     NOT af3_jax's --xla_gpu_cublas_fallback=false: under it XLA finds no Triton config for ColabFold's sub-batch tail GEMM fusions
#     (a dynamic_slice of [3,199,128] fused into the GEMM) and every stock run fails at compile (smoke 1, job 50422042); XLA's default
#     (fallback on) lets those few fusions use hipBLASLt. No flags at all segfaults (rc 139, smoke 1 A7). CFS_XLA_FLAGS replaces the set;
#   - multi-GCD: --xla_gpu_unsupported_use_all_reduce_one_shot_kernel=false (XLA's own small all-reduce kernel spins forever on gfx90a for any
#     psum <= 4 KiB; with the flag every psum goes to RCCL). Inert on one GCD;
#   - launch serialisation: CFS_SERIALISE=copy (default: AMD_SERIALIZE_COPY=3, copy-only, as af3_jax's N2) | blocking (HIP_LAUNCH_BLOCKING=1)
#     | none. ColabFold is launch-bound (4-row sub-batch loops): on 1BRS blocking costs 3.3x against copy-only (smoke 2: 13.5 s vs 4.1 s per
#     warm model); unserialised ran (3.6 s) on that input.
W=${CFS_W:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}   # the tree: the directory above scripts/ (natively colabfold-setonix, in the image /opt/colabfold)
export CFS_W=$W
module use /software/setonix/unsupported >/dev/null 2>&1
module load rocm/7.2.4 >/dev/null 2>&1
export PATH=$W/scripts/bin:$W/venv/bin:$PATH                       # the nvidia-smi shim first (the kit's GPU census)
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$W/xdgcache} MPLCONFIGDIR=${MPLCONFIGDIR:-$W/tmp/mpl} PIP_CACHE_DIR=${PIP_CACHE_DIR:-$W/pipcache}
export MIOPEN_USER_DB_PATH=${MIOPEN_USER_DB_PATH:-$W/cache/miopen} MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_CUSTOM_CACHE_DIR:-$W/cache/miopen}
mkdir -p "$MPLCONFIGDIR" "$MIOPEN_USER_DB_PATH" 2>/dev/null
export JAX_PLATFORMS=${JAX_PLATFORMS:-rocm,cpu}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95} TF_FORCE_UNIFIED_MEMORY=${TF_FORCE_UNIFIED_MEMORY:-0}
unset XLA_CLIENT_MEM_FRACTION
export PYTHONHASHSEED=${PYTHONHASHSEED:-0} PYTHONUNBUFFERED=1
export JAX_PALLAS_USE_MOSAIC_GPU=${JAX_PALLAS_USE_MOSAIC_GPU:-0}
export XLA_FLAGS=${CFS_XLA_FLAGS-"--xla_gpu_enable_command_buffer= --xla_disable_hlo_passes=triton-softmax-rewriter --xla_gpu_unsupported_use_all_reduce_one_shot_kernel=false"}${CFS_XLA_EXTRA:+ $CFS_XLA_EXTRA}
case "${CFS_SERIALISE:-copy}" in
  blocking) export HIP_LAUNCH_BLOCKING=1; unset AMD_SERIALIZE_COPY ;;
  copy)     unset HIP_LAUNCH_BLOCKING; export AMD_SERIALIZE_COPY=3 ;;
  none)     unset HIP_LAUNCH_BLOCKING AMD_SERIALIZE_COPY ;;
  *) echo "cf-env.sh: CFS_SERIALISE=${CFS_SERIALISE} is not blocking|copy|none" >&2; return 2 2>/dev/null || exit 2 ;;
esac
export COLABFOLD_OPT_DATA_DIR=${COLABFOLD_OPT_DATA_DIR:-/scratch/references/colabfold_jun2026/database/alphafold2_multimer_v3}
export COLABFOLD_OPT_JIT_ROOT=${COLABFOLD_OPT_JIT_ROOT:-$W/cache/jit}
export MODEL_OPT=${MODEL_OPT:-$W/kitpin}                          # the re-pinned tree (tools/repin163.py): ColabFold 1.6.3 as stock
export OPT_CORE_PALLAS_CC=${OPT_CORE_PALLAS_CC:-8.0} CFS_SHIM_CC=${CFS_SHIM_CC:-8.0}   # the provider column gfx90a is served from: the A100's (boot patch, shim; smoke 2)
