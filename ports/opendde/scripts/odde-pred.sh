#!/bin/bash
# odde-pred.sh [--stock] [--n_gpu P] -i QUERY.json -o OUT [opendde_opt pred / upstream options, e.g. --seeds 101 --det 1]
#
# ONE OpenDDE prediction on the kit's `big` line: one GCD (with the kit's host offload) or, with --n_gpu 2|4|8, row-sharded over that
# many GCDs of one node by the kit's own launcher. Two lines:
#   patched (default)  the native port's scripts/predict_patched7.sh: stock OpenDDE plus eight opt-in fixes, each an environment
#                      setting or a monkeypatch installed through the kit's own opt_core.autoload.patch_attr_at_import (patches/):
#                        v2 dense-window construction (byte-identical), explicit triangle-attention blocks (32 rows x 256 queries),
#                        ROWPAIR_TRIMUL_SUB=128, the native allocator (expandable_segments:False), the A-block threshold 0.60,
#                        ROWPAIR_TRIATT_ROWBLOCK=256, query-blocked stock attention above 4 GiB of logits, row-blocked APB pair bias
#                        above an 8 GiB shard. At 4,096 tokens on 8 GCDs it is 2.45x stock with the structure inside run-to-run
#                        noise; it completes 8,192 and 9,216 tokens (9,216 needs the node's whole memory: sbatch --mem=0).
#   stock (--stock)    the native port's scripts/predict.sh: the kit's big line as shipped, with only what gfx90a needs.
# On both: upstream's torch triangle kernels by stated flag (no cuEquivariance on ROCm), the sharded triangle-attention core as torch
# statements, the four levers whose kernels cannot run on gfx90a composed out by name (ODDE_LEVERS_OFF, below), and --allow-partial
# (the patched line runs without the kit's mandatory allocator and is therefore reported "partial", by name; rc stays 0).
# Nothing else is forced: --sample, --use_msa, --use_template, --cycle and --step keep the kit's defaults unless given.
set -uo pipefail
D=${ODDE_ROOT:-/opt/opendde}

line=patched P=1
args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --stock)    line=stock; shift ;;
    --patched)  line=patched; shift ;;
    --n_gpu)    P=${2:?--n_gpu needs a value}; shift 2 ;;
    --n_gpu=*)  P=${1#--n_gpu=}; shift ;;
    --mode|--mode=*)
      echo "[odde-pred] --mode is fixed to big here (the line this port validated)" >&2; exit 2 ;;
    *)          args+=("$1"); shift ;;
  esac
done
case "$P" in 1|2|4|8) ;; *) echo "[odde-pred] --n_gpu must be 1, 2, 4 or 8 (got $P)" >&2; exit 2 ;; esac

if [ -z "${MODEL_OPT:-}" ]; then
  # shellcheck source=/dev/null
  . "$D/scripts/odde-env.sh" > /dev/null || exit 97
fi

# --- the patched line: predict_patched7.sh's settings, value for value -----------------------------------------------------
if [ "$line" = patched ]; then
  export PYTHONPATH="$D/patches/hook7:$PYTHONPATH"     # hook7 -> hook6 -> ... -> hook2 -> the per-rank hook, each gated by its flag
  export ODDE_TRUNK_PATCH=1                             # v2: both dense N_atom^2 window tensors built directly (byte-identical)
  export ODDE_CHUNK_FLOOR="${ODDE_CHUNK_FLOOR:-0}"      # off: the explicit blocks set the kit's blocks; the floor broke the refiner
  export ODDE_TP_BLOCKS=1                               # the kit's triangle-attention blocks set explicitly, not from stock's chunk
  export ODDE_TP_ATTN_ROWS="${ODDE_TP_ATTN_ROWS:-32}"
  export ODDE_TP_TRIATT_QBLOCK="${ODDE_TP_TRIATT_QBLOCK:-256}"
  export ODDE_TP_PROGRESS="${ODDE_TP_PROGRESS:-1}"      # rank 0: a progress line every ODDE_TP_PROGRESS_EVERY row batches
  export ODDE_NATIVE_ALLOC=1                            # expandable_segments:False through the kit's own two writers
  export ROWPAIR_TRIMUL_SUB="${ROWPAIR_TRIMUL_SUB:-128}" # the trimul b sub-block (the default table is keyed on N only)
  export ODDE_ABLOCK_WHOLE_FRAC="${ODDE_ABLOCK_WHOLE_FRAC:-0.60}"  # A plane whole only below this u/free (kit 0.80; validated ~0.72)
  export ROWPAIR_TRIATT_ROWBLOCK="${ROWPAIR_TRIATT_ROWBLOCK:-256}"  # triangle-bias LayerNorm + projection in row blocks
  export ODDE_ATTN_QBLOCK="${ODDE_ATTN_QBLOCK:-1}"      # stock attention in query blocks above MIN GiB of logits
  export ODDE_ATTN_QBLOCK_MIN_GIB="${ODDE_ATTN_QBLOCK_MIN_GIB:-4}"
  export ODDE_ATTN_QBLOCK_GIB="${ODDE_ATTN_QBLOCK_GIB:-1}"
  export ODDE_APB_ROWS_PATCH="${ODDE_APB_ROWS_PATCH:-1}" # APB pair bias in row blocks above MIN GiB of shard
  export ODDE_APB_ROWS="${ODDE_APB_ROWS:-256}"
  export ODDE_APB_ROWS_MIN_GIB="${ODDE_APB_ROWS_MIN_GIB:-8}"
fi

# --- both lines ------------------------------------------------------------------------------------------------------------
# Optional page-cache warm-up of torch's libraries (natively needed against Lustre; in the image they are read from the SIF).
if [ "${ODDE_PREWARM:-0}" = 1 ]; then
  t0=$SECONDS
  cat "$D"/venv/lib/python3.12/site-packages/torch/lib/*.so > /dev/null 2>&1
  echo "[odde-pred] pre-warm: $((SECONDS - t0)) s" >&2
fi

# Upstream's own kernel flags, so the kit's kernel levers step aside by name and torch's triangle statements run (ODDE_KERNELS=auto
# lets the levers try: a kernel that cannot build on gfx90a then ends the run by name).
KFLAGS=()
case "${ODDE_KERNELS:-torch}" in
  torch) KFLAGS=(--triatt_kernel torch --trimul_kernel torch) ;;
  auto)  KFLAGS=() ;;
esac
# The sharded triangle-attention core: its opt-out is this export (the lever lives only on BIG_TP), torch statements by default.
export ROWPAIR_TRIATT_CORE="${ODDE_TRIATT_CORE:-torch}"
# The levers whose kernels cannot run on gfx90a: the apb sampler kernels request tf32 (an NVIDIA-only operand mode) through a
# capability table that takes gfx90a's (9,0) for an H100, and atom_fused / cond_dedupe sit on them. ODDE_LEVERS_OFF= (empty) clears it.
LEVERS_OFF="${ODDE_LEVERS_OFF-dit_attn_apb,atom_attn_apb,atom_fused,cond_dedupe}"
[ -n "$LEVERS_OFF" ] && export MODEL_OPT_LEVERS_OFF="$LEVERS_OFF"
# A cross-rank hang becomes a named refusal on every rank.
[ "$P" -gt 1 ] && export ROWPAIR_P2P_CHECK="${ROWPAIR_P2P_CHECK:-1}"

echo "[odde-pred] line=$line n_gpu=$P kernels=${ODDE_KERNELS:-torch} levers_off=${MODEL_OPT_LEVERS_OFF:-<none>}" \
     "HIP_LAUNCH_BLOCKING=$HIP_LAUNCH_BLOCKING weights=$OPENDDE_ROOT_DIR ---- start $(date)" >&2
t0=$SECONDS
python -m opendde_opt pred --mode big --n_gpu "$P" --allow-partial ${KFLAGS[@]+"${KFLAGS[@]}"} ${args[@]+"${args[@]}"}
RC=$?
echo "[odde-pred] ---- end $(date) rc=$RC elapsed=$((SECONDS - t0))s" >&2
echo "ODDE_RESULT line=$line P=$P rc=$RC elapsed_s=$((SECONDS - t0))"
exit $RC
