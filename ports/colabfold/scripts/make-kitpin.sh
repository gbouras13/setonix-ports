#!/bin/bash
# make-kitpin.sh -- build the re-pinned tree $W/kitpin (tools/repin163.py; runs Python: inside a job, or the image build). The two PyPI
# wheels must already be in $W/wheels (login node: pip download --no-deps colabfold==1.6.3 alphafold-colabfold==2.3.20 -d $W/wheels).
# CFS_JIT_DEFAULT: the compile-cache root written into configs/mi250x.env as its default (natively $W/cache/jit; the image passes a
# per-user directory under TMPDIR, since the image is read-only).
set -euo pipefail
W=${CFS_W:-$(cd "$(dirname "$0")/.." && pwd)}   # the tree: the directory above scripts/
JIT=${CFS_JIT_DEFAULT:-$W/cache/jit}
REPIN=$W/tools/repin163.py; [ -f "$REPIN" ] || REPIN=$W/scripts/repin163.py
rm -rf "$W/kitpin.new"
STOCK_WHEELS_DIR=$W/wheels "$W/venv/bin/python" "$REPIN" "$W/kit/colabfold" "$W/kitpin.new"
sed -e 's|^# H100 (compute capability 9.0) deployment parameters; configs/a100.env is the A100 file of the same shape.|# MI250X GCD (gfx90a) deployment parameters of the Setonix port (af3-setonix/colabfold/scripts/make-kitpin.sh): the kit'"'"'s configs/h100.env with two defaults changed (target GPU, cache root).|' \
    -e 's|MODEL_OPT_TARGET_GPU:-H100|MODEL_OPT_TARGET_GPU:-MI250X|' \
    -e "s|COLABFOLD_OPT_JIT_ROOT:-\${HOME:-/root}/.cache/colabfold_opt/jit|COLABFOLD_OPT_JIT_ROOT:-$JIT|" \
    "$W/kit/colabfold/configs/h100.env" > "$W/kitpin.new/configs/mi250x.env"
grep -q "MI250X" "$W/kitpin.new/configs/mi250x.env" && grep -q -F "$JIT" "$W/kitpin.new/configs/mi250x.env"
rm -rf "$W/kitpin"; mv "$W/kitpin.new" "$W/kitpin"
echo "KITPIN_OK $W/kitpin"
