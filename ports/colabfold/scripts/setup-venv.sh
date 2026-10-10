#!/bin/bash
# setup-venv.sh -- the ColabFold 1.6.3 venv for the colabfold kit port (Setonix MI250X, ROCm 7.2.4). LOGIN NODE: downloads only
# (pip); every check that runs Python beyond pip goes in slurm/setup-check.sbatch (debug partition).
#   stock    = colabfold[alphafold-minus-jax]==1.6.3 (pulls alphafold-colabfold 2.3.20, dm-haiku 0.0.17) -- the user's choice (2026-10-05),
#              instead of the kit's pinned 1.6.1 / 2.3.13
#   jax      = 0.10.2 + jax-rocm7-plugin/pjrt 0.10.2: the stack validated in the af3_jax port (0.11.2's plugin SIGBUSes on this host)
#   python   = Cray 3.12.12 (plain venv; no conda: ~/.condarc's shared pkgs cache is corrupt)
#   kit      = common/opt_core + colabfold/opt editable, --no-deps (the kit's own `run.sh install` step minus its 1.6.1 pin check)
# Fallback agreed with the user: dm-haiku 0.0.16 if 0.0.17 does not run on jax 0.10.2 (set CFS_HAIKU=0.0.16).
set -euo pipefail
W=${CFS_W:-$(cd "$(dirname "$0")/.." && pwd)}   # the tree: the directory above scripts/
export PIP_CACHE_DIR=$W/pipcache TMPDIR=$W/tmp XDG_CACHE_HOME=$W/xdgcache PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$XDG_CACHE_HOME" "$W/logs"
PY=/opt/cray/pe/python/3.12.12/bin/python3.12
[ -x "$W/venv/bin/python" ] || "$PY" -m venv "$W/venv"
V=$W/venv/bin
"$V/python" -m pip --version
JAX=(jax==0.10.2 jaxlib==0.10.2 jax-rocm7-plugin==0.10.2 jax-rocm7-pjrt==0.10.2)
"$V/python" -m pip install "colabfold[alphafold-minus-jax]==1.6.3" "${JAX[@]}" pytest
if [ -n "${CFS_HAIKU:-}" ]; then "$V/python" -m pip install --no-deps "dm-haiku==$CFS_HAIKU"; fi
"$V/python" -m pip install --no-deps -e "$W/kit/common/opt_core" -e "$W/kit/colabfold/opt"
"$V/python" -m pip freeze --all > "$W/logs/freeze-venv.txt"
"$V/python" -m pip check || echo "PIP_CHECK_NONZERO (see above)"
echo "venv files: $(find "$W/venv" | wc -l)"
echo SETUP_VENV_DONE
