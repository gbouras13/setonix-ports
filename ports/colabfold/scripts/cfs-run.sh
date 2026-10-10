#!/bin/bash
# cfs-run.sh -- the kit's own run.sh, from the re-pinned tree ($MODEL_OPT = $W/kitpin: ColabFold 1.6.3 as stock), with the port's config:
#   cfs-run.sh pred|check|warm --mode off|exact|fast|big [--n_gpu P] [<input> <results> colabfold_batch options...]
# - Launch serialisation by size unless the caller set CFS_SERIALISE: copy-only (AMD_SERIALIZE_COPY=3) up to CFS_COPY_MAX_TOKENS residues
#   (every copy of every chain; scripts/cfs_tokens.py), blocking (HIP_LAUNCH_BLOCKING=1) above. Copy-only is 1.35-3.3x faster but segfaults
#   on the first model call from 2,695 tokens (ladders 1, jobs 50437494 / 50437941); it passed at 2,310 (stock and fast).
# - big --n_gpu P > 1: adds colabfold_batch's own `--compile-mode full` unless the caller passed a --compile-mode. ColabFold 1.6.3's default
#   (`tuned`) passes compiler_options to jax.jit, and the kit's row-shard recipe wraps that jit in its own sharded jit: jax 0.10 refuses
#   compiler_options on a nested jit ("can only be passed to top-level jax.jit", valid 1). `full` passes none (and ran as fast as `tuned`, smoke 2).
# - Kit modes (exact | fast | big) also get the gfx90a boot patch (scripts/boot on PYTHONPATH + CFS_GFX90A=1: opt_core's provider reads gfx90a
#   as the A100 column 8.0; the reviewed alphafold-colabfold 2.3.20 pin entry for big --n_gpu) and the port's lever drops (MODEL_OPT_LEVERS_OFF,
#   the kit's own ablation switch, from scripts/cfs-recipes.sh unless the caller set it). --mode off is the kit's stock route, untouched.
set -u
W=${CFS_W:-$(cd "$(dirname "$0")/.." && pwd)}   # the tree: the directory above scripts/
MODE=""; prev=""; INPUT=""; CMD=""; NGPU=1; HAVE_CM=0
for a in "$@"; do
  case "$a" in --n_gpu=*) NGPU=${a#--n_gpu=} ;; --compile-mode|--compile-mode=*) HAVE_CM=1 ;; esac
  [ "$prev" = "--n_gpu" ] && NGPU=$a
  case "$a" in --mode=*) MODE=${a#--mode=} ;; esac
  [ "$prev" = "--mode" ] && MODE=$a
  if [ -z "$CMD" ] && [[ "$a" != -* ]]; then CMD=$a
  elif [ -z "$INPUT" ] && [ -n "$CMD" ] && [[ "$a" != -* ]] && [[ "$prev" != --* ]] && [ -e "$a" ]; then INPUT=$a; fi
  prev=$a
done
if [ -z "${CFS_SERIALISE:-}" ]; then
  CFS_COPY_MAX_TOKENS=${CFS_COPY_MAX_TOKENS:-2310}
  TOK=0; [ -n "$INPUT" ] && TOK=$(python3 "$W/scripts/cfs_tokens.py" "$INPUT" 2>/dev/null || echo 0)
  if [ "${TOK:-0}" -gt 0 ] && [ "${TOK:-0}" -le "$CFS_COPY_MAX_TOKENS" ]; then export CFS_SERIALISE=copy; else export CFS_SERIALISE=blocking; fi
  echo "[cfs] input=${INPUT:-none} tokens=$TOK -> serialise=$CFS_SERIALISE (copy-only up to $CFS_COPY_MAX_TOKENS tokens; set CFS_SERIALISE to override)" >&2
fi
source "$W/scripts/cf-env.sh" || exit $?
if [ -n "$MODE" ] && [ "$MODE" != off ]; then
  export PYTHONPATH="$W/scripts/boot${PYTHONPATH:+:$PYTHONPATH}" CFS_GFX90A=1
  if [ -z "${MODEL_OPT_LEVERS_OFF+x}" ] && [ -f "$W/scripts/cfs-recipes.sh" ]; then
    source "$W/scripts/cfs-recipes.sh"
    v="CFS_LEVERS_OFF_${MODE^^}"; [ -n "${!v:-}" ] && export MODEL_OPT_LEVERS_OFF="${!v}"
  fi
  echo "[cfs] mode=$MODE MODEL_OPT=$MODEL_OPT MODEL_OPT_LEVERS_OFF=${MODEL_OPT_LEVERS_OFF:-} OPT_CORE_PALLAS_CC=$OPT_CORE_PALLAS_CC serialise=$CFS_SERIALISE" >&2
fi
EXTRA=()
if [ "$CMD" = pred ] && [ "${NGPU:-1}" -gt 1 ] 2>/dev/null && [ "$HAVE_CM" = 0 ]; then
  EXTRA=(--compile-mode full); echo "[cfs] --n_gpu $NGPU: adding colabfold_batch --compile-mode full (1.6.3's tuned compiler_options cannot sit under the sharded jit)" >&2
fi
exec bash "$MODEL_OPT/run.sh" --config mi250x "$@" ${EXTRA[@]+"${EXTRA[@]}"}
