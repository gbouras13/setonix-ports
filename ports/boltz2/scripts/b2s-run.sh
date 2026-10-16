#!/bin/bash
# b2s-run.sh <verb> --mode <m> [args...] -- the kit's run.sh route with its last line replaced:
#
#   run.sh:      check_pins.py --quiet, then `exec python -m boltz2_opt <verb> --mode <m> ...`
#   b2s-run.sh:  check_pins.py --quiet, then `exec python scripts/b2s_launch.py <verb> --mode <m> ...`
#
# b2s_launch.py applies the opt-in mode-table patches (B2S_KERNELS, B2S_CARD_DROPS) in the parent process and then calls the
# kit's own entry (boltz2_opt.__main__.main: core pin gate, module check, cli). With neither variable set it is the kit's
# route exactly. --config is not taken: source scripts/bz-env.sh first (it sets what configs/<card>.env sets, for gfx90a).
# The stock route (--mode off) is the kit's own run.sh; use that for stock baselines.
set -euo pipefail
D=${B2S_ROOT:-/scratch/pawsey1018/gbouras/boltz2-setonix}
: "${MODEL_OPT:?source scripts/bz-env.sh first}"
env -u BOLTZ2_OPT python -I "$MODEL_OPT/stock/check_pins.py" --quiet || { echo "b2s-run: boltz is not installed at the pin (stock/PINS.json)" >&2; exit 3; }
exec python "$D/scripts/b2s_launch.py" "$@"
