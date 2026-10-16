#!/bin/bash
# ptx-pred.sh -- one Protenix v2 prediction with the gfx90a recipe for its route (PROTENIX_PORT.md, "The working configuration"):
#
#   bash scripts/ptx-pred.sh [--mode big|off|stock] [--n_gpu P] -i Q.json -o OUT [stock options, e.g. --seeds 101 --use_msa false]
#
# It sources scripts/ptx-env.sh unless that ran already (the kit's paths, PROTENIX_OPT_FORCE=1, LAYERNORM_TYPE=torch, three levers off,
# HIP_LAUNCH_BLOCKING=1, the sharded line's words), checks that the seven pinned weight files are under PROTENIX_ROOT_DIR, and runs:
#   big (default)  python -m protenix_opt pred --mode big: the kit's memory line, with transition_core dropped by name as well (its kernel
#                  builds on gfx90a and returns NaN with exit 0: PTX inline assembly, job 49905402)
#     one GCD      + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False. The kit's default strands freed memory on ROCm 7.2: four of the
#                  one-GCD ladder's eight sizes failed on it and complete without it (jobs 50229219, 50233809). With deterministic
#                  algorithms this line is stock's arithmetic bit for bit (jobs 50030425, 50379609)
#     --n_gpu P    + PXS_TRIATT_CHUNK_CAP=4: stock's triangle-attention logits, [rows, 8, N, N] per call, bounded at 4 rows (hooks/
#                  setonix_chunkcap.py), as in every native sharded run. ptx-env.sh adds the engine's DiT attention, opt_core's fused
#                  triangle attention and two RCCL settings. The kit's own allocator setting stays: the 12,288-token run completed on it
#   off            python -m protenix_opt pred --mode off: stock Protenix in the kit's own clean, self-proving subprocess
#   stock          stock Protenix's own CLI (runner.batch_inference.protenix_cli) with PROTENIX_OPT unset, so the kit applies nothing: the
#                  route for deterministic runs (PXS_DETERMINISTIC=1), which --mode off refuses (it finds torch already imported)
# Every route gets --model_name protenix-v2 --triatt_kernel torch --trimul_kernel torch unless the command gives them: the one pinned
# checkpoint, and no cuEquivariance on ROCm. A value already in the environment wins over each default above: they are defaults, not
# locks. The kit's exact and fast modes are not validated on gfx90a; `protenix kit pred --mode ...` runs them without a recipe.
# --n_gpu P puts its ranks on the first P GCDs the process sees (cuda:LOCAL_RANK): give concurrent runs their own ROCR_VISIBLE_DEVICES.
set -euo pipefail
D=${PXS_ROOT:-/scratch/pawsey1018/gbouras/protenix-setonix}
if [ -z "${MODEL_OPT:-}" ]; then
  # shellcheck source=/dev/null
  . "$D/scripts/ptx-env.sh" > /dev/null || exit 97
fi
say() { echo "[ptx-pred] $*" >&2; }

mode=big n_gpu=1 have=" " args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --mode|--n_gpu)
      [ $# -ge 2 ] || { say "$1 needs a value"; exit 2; }
      if [ "$1" = --mode ]; then mode=$2; else n_gpu=$2; fi
      shift 2; continue ;;
    --mode=*)  mode=${1#--mode=}; shift; continue ;;
    --n_gpu=*) n_gpu=${1#--n_gpu=}; shift; continue ;;
    --model_name|--model_name=*|--triatt_kernel|--triatt_kernel=*|--trimul_kernel|--trimul_kernel=*) have="$have${1%%=*} " ;;
  esac
  args+=("$1"); shift
done
case "$n_gpu" in ''|*[!0-9]*|0) say "--n_gpu must be a GPU count, 1 or more (got '$n_gpu')"; exit 2 ;; esac
case "$have" in *" --model_name "*) ;; *) args+=(--model_name protenix-v2) ;; esac
case "$have" in *" --triatt_kernel "*) ;; *) args+=(--triatt_kernel torch) ;; esac
case "$have" in *" --trimul_kernel "*) ;; *) args+=(--trimul_kernel torch) ;; esac

# The seven files of stock/PINS.json, by name only (the kit checks their digests itself, on its WEIGHTS line). PROTENIX_ROOT_FROZEN is
# not set (ptx-env.sh says why), so stock would otherwise start downloading whatever is absent, mid-job. -S: no site, so none of the
# start-up hooks a PXS_* setting arms runs for this.
missing=$(python -I -S - "$MODEL_OPT/stock/PINS.json" "${PROTENIX_ROOT_DIR:-}" <<'PY'
import json, os, sys
pins, root = json.load(open(sys.argv[1])), sys.argv[2]
want = [pins["checkpoint"]["file"]] + list(pins["data_caches"]["files"])
print(" ".join(f for f in want if not root or not os.path.isfile(os.path.join(root, f))))
PY
) || { say "could not read $MODEL_OPT/stock/PINS.json"; exit 4; }
if [ -n "$missing" ]; then
  say "weights incomplete under '${PROTENIX_ROOT_DIR:-}': missing $missing -- fetch them first (protenix fetch-weights DIR)"
  exit 4
fi

STOCK_CLI='import sys; from runner.batch_inference import protenix_cli; sys.argv = ["protenix"] + sys.argv[1:]; protenix_cli()'
case "$mode" in
  big)
    case ",${MODEL_OPT_LEVERS_OFF:-}," in
      *,transition_core,*) ;;
      *) export MODEL_OPT_LEVERS_OFF="${MODEL_OPT_LEVERS_OFF:+$MODEL_OPT_LEVERS_OFF,}transition_core" ;;
    esac
    if [ "$n_gpu" -gt 1 ]; then
      export PXS_TRIATT_CHUNK_CAP="${PXS_TRIATT_CHUNK_CAP:-4}"
      line=(--mode big --n_gpu "$n_gpu")
    else
      export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
      line=(--mode big)
    fi
    say "route=big n_gpu=$n_gpu LEVERS_OFF=$MODEL_OPT_LEVERS_OFF PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-kit default}" \
        "PXS_TRIATT_CHUNK_CAP=${PXS_TRIATT_CHUNK_CAP:-stock} TRIATT_CORE=${ROWPAIR_TRIATT_CORE:-} DIT_ROWS=${ROWPAIR_DIT_ROWS:-}" \
        "weights=$PROTENIX_ROOT_DIR"
    exec python -m protenix_opt pred "${line[@]}" "${args[@]}" ;;
  off)
    unset MODEL_OPT_LEVERS_OFF                     # the stock route takes no lever list (validate.sbatch's V1_off, job 49983466)
    line=(--mode off); [ "$n_gpu" -gt 1 ] && line+=(--n_gpu "$n_gpu")   # the kit refuses P > 1 under off by name
    say "route=off (stock, in the kit's clean subprocess) weights=$PROTENIX_ROOT_DIR"
    exec python -m protenix_opt pred "${line[@]}" "${args[@]}" ;;
  stock)
    [ "$n_gpu" = 1 ] || { say "the stock CLI runs on one GPU: drop --n_gpu, or use the big line"; exit 2; }
    unset MODEL_OPT_LEVERS_OFF PROTENIX_OPT
    say "route=stock (stock Protenix's own CLI; nothing of the kit applied) deterministic=${PXS_DETERMINISTIC:-0} weights=$PROTENIX_ROOT_DIR"
    exec python -c "$STOCK_CLI" pred "${args[@]}" ;;
  exact|fast)
    say "--mode $mode is not validated on gfx90a by this port (only big and off are); 'protenix kit pred --mode $mode ...' runs it unguarded"
    exit 2 ;;
  *)
    say "unknown mode '$mode' (big, off, stock)"; exit 2 ;;
esac
