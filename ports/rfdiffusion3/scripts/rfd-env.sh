#!/bin/bash
# rfd-env.sh -- the RFdiffusion3 image's environment (the native port's scripts/rfd-env.sh, with the image's paths).
#
#   source scripts/rfd-env.sh [stock|kit]      default kit, the interpreter that runs exact | fast
#
# This kit is a FILE OVERLAY on the installed rfd3 package, so it needs two interpreters that never share a site-packages:
# venv-stock is pristine and runs `--mode off` under `python -I`; venv-kit carries the ten-file overlay and runs exact / fast.
# It does not replace configs/h100.env: every value that file sets is `${VALUE:-default}`, so pre-setting the machine's words
# here lets `run.sh --config h100` fill the rest and run its own probes.
D=${RFD_ROOT:-/opt/rfdiffusion3}
WHICH=${1:-${RFD_VENV:-kit}}
case "$WHICH" in stock|kit) ;; *) echo "[rfd] usage: source scripts/rfd-env.sh [stock|kit]" >&2; return 2 2>/dev/null || exit 2 ;; esac
for v in venv-stock venv-kit; do
  if [ ! -x "$D/$v/bin/python" ]; then echo "[rfd] NO VENV at $D/$v" >&2; return 97 2>/dev/null || exit 97; fi
done

export VIRTUAL_ENV="$D/venv-$WHICH"
export PATH="$D/scripts/bin:$VIRTUAL_ENV/bin:${PATH}"

# --- what configs/h100.env would otherwise fill ---------------------------------------------------------------------------
export MODEL_OPT="${MODEL_OPT:-$D/kit/rfdiffusion3}"
export MODEL_OPT_TARGET_GPU="${MODEL_OPT_TARGET_GPU:-MI250X}"      # a label; `check` reports the machine's own card against it
# jit_cache_key() raises on ROCm (torch.version.cuda is unset) and configs/h100.env exits 2 rather than guess: preset it.
export MODEL_OPT_STACK_KEY="${MODEL_OPT_STACK_KEY:-torch2.14.0-rocm7.2-gfx90a}"
export RFDIFFUSION3_STOCK_PYTHON="${RFDIFFUSION3_STOCK_PYTHON:-$D/venv-stock/bin/python}"
export RFD3_CKPT="${RFD3_CKPT:-/weights/rfd3/rfd3_latest.ckpt}"     # mounted; the image ships no checkpoint
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

# --- writable scratch, outside the read-only image and outside $HOME ------------------------------------------------------
: "${RFD_LOCAL:=${TMPDIR:-/tmp}/rfd-$(id -u)}"
mkdir -p "$RFD_LOCAL"/{tmp,xdg,miopen} 2>/dev/null || true
export RFD_LOCAL TMPDIR="$RFD_LOCAL/tmp" XDG_CACHE_HOME="$RFD_LOCAL/xdg"
export MIOPEN_USER_DB_PATH="$RFD_LOCAL/miopen" MIOPEN_CUSTOM_CACHE_DIR="$RFD_LOCAL/miopen"
export RFDIFFUSION3_OPT_CACHE="${RFDIFFUSION3_OPT_CACHE:-$RFD_LOCAL/ckptdigest}"
export MODEL_OPT_JIT_ROOT="${MODEL_OPT_JIT_ROOT:-$RFD_LOCAL/jit}"
# configs/h100.env keys these three under MODEL_OPT_JIT_ROOT, but only run.sh sources it: a route that calls upstream's own
# command line directly would otherwise write to $HOME/.triton, where Setonix's home quota stops it.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$MODEL_OPT_JIT_ROOT/$MODEL_OPT_STACK_KEY/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$MODEL_OPT_JIT_ROOT/$MODEL_OPT_STACK_KEY/torch_extensions}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$MODEL_OPT_JIT_ROOT/$MODEL_OPT_STACK_KEY/inductor}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" 2>/dev/null || true

# --- the two gfx90a settings this port measured ---------------------------------------------------------------------------
# 1. HIP graph packet capture OFF. RFD3_CUDAGRAPH captures the denoiser step and replays it once per timestep; ROCm 7.2's default
#    packet capture corrupts the host heap doing that (seen in two other codebases on this machine).
export DEBUG_CLR_GRAPH_PACKET_CAPTURE="${RFD_GRAPH_PACKET_CAPTURE:-${DEBUG_CLR_GRAPH_PACKET_CAPTURE:-0}}"
# 2. The fused transition steps aside on this card. Its row table (FZT_ROWS_BY_CC) has a row for compute capability 8.0 only and
#    fuses EVERY row count at 9.0 -- which is what gfx90a reports -- on the grounds that cuBLAS keeps one summation order there.
#    This card's GEMMs are hipBLASLt's, and with the lever on, `exact` is NOT stock's designs (one 20.2 A away). scripts/boot
#    gives capability 9.0 an empty served-row range in memory, so the lever takes the stock body BY NAME -- a route the kit
#    defines for itself. RFD_BOOT= (empty) turns it off; the kit's own APPLIED line reports `fused_calls=0` when it is on.
export RFD_BOOT="${RFD_BOOT-fzt_rows_off}"
case ":${PYTHONPATH:-}:" in
  *":$D/scripts/boot:"*) ;;
  *) export PYTHONPATH="$D/scripts/boot${PYTHONPATH:+:$PYTHONPATH}" ;;
esac

# --- allocator: left as the kit sets it (expandable_segments:True) --------------------------------------------------------
# Measured both ways on this card: `exact` at diffusion_batch_size 16 runs out of memory identically with True and False, so the
# setting is not the limit -- the roll-out cache and one captured graph pool per design length are (57 GiB against stock's 8).
case "${RFD_ALLOC:-}" in
  True|true)   export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" ;;
  False|false) export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False" ;;
esac

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS:-8}}"
echo "[rfd] venv=$VIRTUAL_ENV kit=$MODEL_OPT ckpt=$RFD3_CKPT boot=${RFD_BOOT:-none} local=$RFD_LOCAL" \
     "GRAPH_PACKET_CAPTURE=$DEBUG_CLR_GRAPH_PACKET_CAPTURE ALLOC=${PYTORCH_CUDA_ALLOC_CONF:-unset(kit default)}" >&2
