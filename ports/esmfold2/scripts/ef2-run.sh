#!/bin/bash
# ef2-run.sh -- the kit's run.sh, verbatim but for TWO lines, for this machine. Same arguments as run.sh:
#
#   bash scripts/ef2-run.sh pred|check|warm --config h100 --variant V [--mode M] ... (source scripts/ef2-env.sh first)
#
# The kit's run.sh refuses every verb on this machine at one line: `python -I stock/check_pins.py --quiet`, whose image-layer half
# requires the CUDA builds of flash-attn / xformers / transformer-engine (stock/PINS.json "image"), which do not exist for ROCm.
# This script writes a copy of the kit's own run.sh to node-local storage with exactly two lines replaced, checks that nothing
# else differs, and runs the copy:
#   1. HERE=...      -> HERE=$MODEL_OPT (the copy does not live in the kit directory; run.sh locates the kit through HERE)
#   2. check_pins    -> scripts/efs_pins.py: the SAME check_pins.py functions; the forks' commit pins still refuse (exit 3),
#                       the image layer is printed as absent and the run goes on (ESMFOLD2_OPT_REQUIRE_FAST_ENV=0 in ef2-env.sh)
# Everything else -- config sourcing, mode / variant rules, HF_HOME refusals, the core gate, the producer census, the exec of
# `python -m esmfold2_opt` -- is the kit's own text. The kit tree is never edited.
set -euo pipefail
D=${EFS_ROOT:-/scratch/pawsey1018/gbouras/esmfold2-setonix}
[ -n "${MODEL_OPT:-}" ] && [ -f "$MODEL_OPT/run.sh" ] || { echo "[efs] ef2-run.sh: MODEL_OPT is not the kit's esmfold2 directory (source scripts/ef2-env.sh)" >&2; exit 2; }
SRC=$MODEL_OPT/run.sh
OUT=${EFS_LOCAL:-${TMPDIR:-/tmp}}/run-efs.$$.sh
PINLINE='python -I "$HERE/stock/check_pins.py" --quiet || {'
grep -c -F "$PINLINE" "$SRC" | grep -qx 1 || { echo "[efs] ef2-run.sh: the kit's run.sh no longer has its one check_pins line -- review this wrapper" >&2; exit 2; }
grep -c '^HERE=' "$SRC" | grep -qx 1 || { echo "[efs] ef2-run.sh: the kit's run.sh no longer has its one HERE= line -- review this wrapper" >&2; exit 2; }
sed -e 's#^HERE=.*#HERE=$MODEL_OPT; export MODEL_OPT=$HERE#' \
    -e "s#python -I \"\$HERE/stock/check_pins.py\" --quiet || {#python -I \"$D/scripts/efs_pins.py\" \"\$HERE\" || {#" \
    "$SRC" > "$OUT"
n=$(diff "$SRC" "$OUT" | grep -c '^>' || true)
[ "$n" = 2 ] || { echo "[efs] ef2-run.sh: expected 2 changed lines against the kit's run.sh, found $n -- not running" >&2; diff "$SRC" "$OUT" >&2 || true; rm -f "$OUT"; exit 2; }
trap 'rm -f "$OUT"' EXIT
bash "$OUT" "$@"
