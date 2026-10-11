#!/bin/bash
# b2s-pred.sh -- one Boltz-2 prediction with the gfx90a recipe for its mode (BOLTZ2_PORT.md, "The working configuration"):
#
#   bash scripts/b2s-pred.sh --mode off|exact|fast|big [--n_gpu P] --input X.yaml --out_dir OUT [upstream options, e.g. --seed 1]
#
# It sources scripts/bz-env.sh unless that ran already, stages this job's BOLTZ_CACHE (scripts/stage-cache.sh: the four weight files
# from $B2S_WEIGHTS, default $B2S_ROOT/weights, and mols/ unpacked on node-local disk), and runs:
#   off           the kit's own run.sh, `pred --mode off ... --no_kernels`: stock boltz in the kit's clean, self-proving subprocess
#   exact         B2S_KERNELS=off B2S_CARD_DROPS=$B2S_DROPS_EXACT                              byte-identical to stock
#   fast          B2S_KERNELS=off B2S_CARD_DROPS=$B2S_DROPS_FAST                               opt_core's flash kernel in the kit's block
#                                                                                               (b2s_launch.py default): 2.1x stock;
#                                                                                               B2S_FAST_PATCH=1 ($B2S_DROPS_FASTP + the
#                                                                                               opt-in row patch): 2.9x. B2S_FAST_FLASH=0
#                                                                                               drops the flash kernel (1.06x / 1.24x)
#   big           B2S_KERNELS=off B2S_CARD_DROPS=$B2S_DROPS_BIGP                               one GCD; scripts/b2s_launch.py makes it the
#                                                                                               flash recipe (B2S_BIG_FLASH=0 opts out)
#   big --n_gpu P B2S_KERNELS=off OF3T_FLASH_IP=ieee B2S_CARD_DROPS=$B2S_DROPS_BIGTP           row-sharded over P GCDs of one node, with
#                                                                                               the speed words (B2S_TP_SPEED=0 opts out)
# No --mode means the kit's default, fast. A B2S_CARD_DROPS, B2S_WORKER_PATCHES or OF3T_FLASH_IP already set wins: the recipes are
# defaults, not locks. The kit's featurized-batch hand-over goes to node-local disk (BOLTZ_OPT_XFER_DIR), never to /scratch, where
# large inputs stalled for hours (P3 round 2). One sharded (--n_gpu) prediction per node: two on one node slowed each other ~70x.
set -euo pipefail
D=${B2S_ROOT:-/scratch/pawsey1018/gbouras/boltz2-setonix}
if [ -z "${MODEL_OPT:-}" ]; then
  # shellcheck source=/dev/null
  . "$D/scripts/bz-env.sh" > /dev/null || exit 97
fi
# shellcheck source=/dev/null
. "$D/scripts/b2s-recipes.sh"

mode=fast n_gpu=1 prev=
for a in "$@"; do
  case "$prev" in --mode) mode=$a ;; --n_gpu) n_gpu=$a ;; esac
  case "$a" in --mode=*) mode=${a#--mode=} ;; --n_gpu=*) n_gpu=${a#--n_gpu=} ;; esac
  prev=$a
done

if [ -z "${BOLTZ_CACHE:-}" ]; then
  BOLTZ_CACHE=$(bash "$D/scripts/stage-cache.sh" | tail -1) || { echo "[b2s-pred] no weights: see stage-cache.sh above" >&2; exit 4; }
  export BOLTZ_CACHE
fi
export BOLTZ_OPT_XFER_DIR="${BOLTZ_OPT_XFER_DIR:-$B2S_LOCAL/xfer}"
mkdir -m 700 -p "$BOLTZ_OPT_XFER_DIR"

case "$mode" in
  off)
    case " $* " in *" --no_kernels "*) ;; *) set -- "$@" --no_kernels ;; esac
    echo "[b2s-pred] mode=off: the kit's run.sh (stock boltz, --no_kernels)  cache=$BOLTZ_CACHE" >&2
    exec bash "$MODEL_OPT/run.sh" pred "$@" ;;
  exact) drops=$B2S_DROPS_EXACT ;;
  fast)
    if [ "${B2S_FAST_PATCH:-0}" = 1 ]; then
      drops=$B2S_DROPS_FASTP
      export B2S_WORKER_PATCHES="${B2S_WORKER_PATCHES:-$B2S_PATCHES_GFX90A}"
    else
      drops=$B2S_DROPS_FAST
    fi ;;
  big)
    if [ "$n_gpu" -gt 1 ] 2>/dev/null; then
      drops=$B2S_DROPS_BIGTP
      export OF3T_FLASH_IP="${OF3T_FLASH_IP:-ieee}"
    else
      drops=$B2S_DROPS_BIGP
    fi ;;
  *) echo "[b2s-pred] unknown mode '$mode' (off, exact, fast, big)" >&2; exit 2 ;;
esac
export B2S_KERNELS="${B2S_KERNELS:-off}" B2S_CARD_DROPS="${B2S_CARD_DROPS:-$drops}"
echo "[b2s-pred] mode=$mode n_gpu=$n_gpu  B2S_KERNELS=$B2S_KERNELS  B2S_WORKER_PATCHES=${B2S_WORKER_PATCHES:-}  OF3T_FLASH_IP=${OF3T_FLASH_IP:-}  cache=$BOLTZ_CACHE" >&2
exec bash "$D/scripts/b2s-run.sh" pred "$@"
