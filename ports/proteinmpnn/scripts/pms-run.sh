#!/bin/bash
# pms-run.sh -- the kit's run.sh (design | check | warm), with `python -m proteinmpnn_opt` replaced by scripts/pms_launch.py (the worker
# routed through scripts/pms_worker_boot.py: PMS_WORKER_PATCHES, default rocm_draw on a ROCm torch).
# Source scripts/pms-env.sh first (it replaces configs/<card>.env and run.sh's JIT-cache keying: MODEL_OPT_JIT_ROOT, TRITON_CACHE_DIR).
# The stock route needs no patch: use the kit's own run.sh for `--mode off` (this works too: the stock route launches no worker).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python -c "import proteinmpnn_opt" 2>/dev/null || { echo "[proteinmpnn-opt] NOT ACTIVE: proteinmpnn_opt is not installed on $(command -v python)" >&2; exit 3; }
python -c "from proteinmpnn_opt import core_gate; core_gate()" >/dev/null || exit 3
exec python "$HERE/pms_launch.py" "$@"
