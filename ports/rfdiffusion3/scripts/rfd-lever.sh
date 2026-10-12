#!/bin/bash
# One lever of the kit, on upstream's own command line, in the kit interpreter.
#
#   bash scripts/rfd-lever.sh <levers|none> <det 0|1> design inputs=<spec> out_dir=<dir> seed=101 [...]
#
# RFD_DET_STRICT=1 additionally makes torch raise on the first op with no deterministic implementation, naming it.
# <levers> is a comma-separated list of RFD3_* switch names, or `none` for the overlay's stock code paths (every switch
# unset: the files take upstream's branch, so this is the bisection's own baseline, measured in the same interpreter).
# The two deterministic variables must be exported BEFORE python imports torch, which is why they are set here.
set -uo pipefail
D=${RFD_ROOT:-/scratch/pawsey1018/gbouras/rfdiffusion-setonix}
L=$1; DET=$2; shift 2
[ "$L" = none ] && L=""
if [ "$DET" = 1 ]; then export CUBLAS_WORKSPACE_CONFIG=":4096:8" FOUNDRY_DET_SCATTER=1; fi
export RFD_DET=$DET RFD_DET_STRICT=${RFD_DET_STRICT:-0}
unset RFDIFFUSION3_OPT RFDIFFUSION3_OPT_DET
exec python "$D/scripts/rfd_lever_run.py" --levers "$L" -- "$@"
