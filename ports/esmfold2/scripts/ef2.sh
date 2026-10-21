#!/bin/bash
# ef2.sh -- run ESMFold2 on a Setonix MI250X GCD by ROUTE, with this machine's settings for that route.
#
#   bash scripts/ef2.sh <route> --variant fast|full_msa|full_nomsa --input X.json --out_dir DIR [--seeds 0 ...any run.sh pred flag]
#
# routes   stock         --mode off (upstream as loaded; no fix needed beyond launch blocking)
#          stock-fused   --mode off --backend fused (upstream's fused Triton backend; needs dotprec)
#          exact         --mode exact: byte-identical to stock-fused (dotprec, smem; t3,t6 off; glue off on the full model)
#          fast          --mode fast  (dotprec, smem; t3,t16,t10,tx off -- glue is exact's alone: fast's set does not carry it).
#                        tx off is this port's fix for fast's NaN above 256 tokens (the shared core's TriMul cell steps aside to
#                        rows that return NaN on gfx90a). Own numerics: median CA RMSD 0.041 A to stock fused (valid4-6); 1.34x exact.
#          big           --mode big   (as fast, no CUDA graphs; the memory mode)
# Sources scripts/ef2-env.sh and scripts/ef2-stage.sh (the weights to the node's NVMe, once per job) when the caller has not.
# HIP_LAUNCH_BLOCKING=1 by default on every route: on ROCm 7.2 every crash measured here (stock under --det 1, exact's third captured
# graph generation, fast's sampler graph replay) was asynchronous and none happened with blocking. EFS_LAUNCH_BLOCKING=0 turns it off.
# exact / fast run with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False (EFS_ALLOC=True|False overrides): the kit exports True on its
# graph lines, and on ROCm 7.2 memory torch unmaps there is never given back, so every input's released graph generation stays held
# (OOM after 39-48 inputs); with False the 48 bench targets fold in one process at flat memory, byte-identical, as fast (valid5).
# big keeps the kit's True (its line refuses any other value).
set -uo pipefail
D=${EFS_ROOT:-/scratch/pawsey1018/gbouras/esmfold2-setonix}
route=${1:-}; shift || true
variant=""
for ((i = 1; i <= $#; i++)); do [ "${!i}" = --variant ] && { j=$((i + 1)); variant=${!j}; }; done
full=0; case "$variant" in full_msa|full_nomsa) full=1 ;; esac
G=""; [ "$full" = 1 ] && G=",glue"
alloc=""
case "$route" in
  stock)       boot="";            off="";              mode=(--mode off) ;;
  stock-fused) boot="dotprec";     off="";              mode=(--mode off --backend fused) ;;
  exact)       boot="dotprec,smem"; off="t3,t6$G";      mode=(--mode exact); alloc=expandable_segments:False ;;
  fast)        boot="dotprec,smem"; off="t3,t16,t10,tx"; mode=(--mode fast); alloc=expandable_segments:False ;;
  big)         boot="dotprec,smem"; off="t3,t16,t10";   mode=(--mode big) ;;
  *) sed -n '2,19p' "$0" >&2; exit 2 ;;
esac
[ -n "${VIRTUAL_ENV:-}" ] && [ "${VIRTUAL_ENV}" = "$D/venv" ] || source "$D/scripts/ef2-env.sh" || exit 1
# EFS_STAGE=0 reads the weights where HF_HOME points (no node-local copy; slow off Lustre: mmap page faults)
[ "${EFS_STAGE:-1}" = 0 ] || { [ -n "${EFS_WEIGHTS_LOCAL:-}" ] && [ -f "${EFS_WEIGHTS_LOCAL}/.staged" ]; } || source "$D/scripts/ef2-stage.sh"
export EFS_BOOT="$boot"
if [ -n "$off" ]; then export MODEL_OPT_LEVERS_OFF="$off"; else unset MODEL_OPT_LEVERS_OFF; fi
[ "${EFS_LAUNCH_BLOCKING:-1}" = 0 ] && unset HIP_LAUNCH_BLOCKING || export HIP_LAUNCH_BLOCKING=1
case "${EFS_ALLOC:-}" in      # EFS_ALLOC=True: the kit's own setting (one long input near the one-GCD ceiling: 1,900 vs 1,800 tokens)
  True|true)   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ;;
  False|false) export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False ;;
  *)           [ -n "$alloc" ] && export PYTORCH_CUDA_ALLOC_CONF="$alloc" ;;
esac
echo "[efs] route=$route variant=${variant:-?} EFS_BOOT=${boot:-none} MODEL_OPT_LEVERS_OFF=${off:-none} HIP_LAUNCH_BLOCKING=${HIP_LAUNCH_BLOCKING:-0} PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-unset}" >&2
exec bash "$D/scripts/ef2-run.sh" pred --config h100 "${mode[@]}" "$@"
