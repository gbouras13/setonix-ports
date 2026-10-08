#!/bin/bash
# af3jax_predict.sh -- AlphaFold 3 (JAX fork; OpenFold3-preview2 weights, or the official AF3 parameters with --model_dir) on Setonix
# MI250X with the validated speed and memory levers,
# the configuration chosen from the input's size (AF3JAX_PERF_PLAN.md, "Next steps", N1). Run it INSIDE a Slurm allocation on a GPU node:
# a whole node for regime C; one GCD is enough for A and B.
#
#   af3jax_predict.sh --input_dir DIR --output_dir DIR [--regime auto|A|B|C|stock1|stock8] [--gcd N] [--bucket N] [--num_recycles R]
#                     [--serialise copy|block] [--cache_dir DIR] [--gemm triton|library|rocblas] [--model_dir DIR] [--dry_run]
#                     [-- extra run_alphafold.py flags]
#
# Weights: by default the OpenFold3-preview2 weights converted to AF3 format ($AF3JAX_W/weights/p2, run with --of3_weights), the weights
# every validation in AF3JAX_PORT.md used. --model_dir DIR (or AF3JAX_MODEL_DIR) runs the official AlphaFold 3 parameters instead: DIR
# holds af3.bin.zst (Google DeepMind's terms of use apply), and the model runs without --of3_weights, i.e. AlphaFold 3's own network.
# The speed setups A/B/C were validated on the OpenFold3 weights; with the AF3 parameters compare --regime stock1 against auto once.
#
# Regimes (validated 2026-09-29/30, AF3JAX_PORT.md):
#   A       one GCD, bucket <= $AF3JAX_A_MAX (default 3,584): the C1 diffusion levers + flash triangle attention
#           (PERF_LEVERS=cond_share,hoist,hoist_logits,sampler_bf16,triattn_flash; 2.65x at 1,664 tokens, 10 recycles; at 1 recycle 2.89x
#           at 3,200 and 3.02x at 3,584, N1b job 50194710). Its peak is stock's: 57.71 GiB of 60.79 at 3,584; 3,712 would not fit.
#   B       one GCD, bucket <= $AF3JAX_B_MAX (default 4,480: 56.48 GiB, N1b; 4,608 also folds, with 1.15 GiB to spare): flash + the kit's
#           memory levers (PERF_MEM_LEVERS=cond_shard,samples_per_pass)
#           + sampler_bf16. The hoist is off (its resident logits and pair conditioning are ~30 GiB at 3,968 tokens) and so is cond_share
#           (the kit's one-sample-per-pass sampler re-implements the loop and never calls it). Lifts the one-GCD limit past 4,235 residues;
#           slower than A (1.31x at 3,584 against A's 3.02x), so it starts where A stops.
#   C       8 GCDs above that: row-sharded (rowpair_launch.py) + flash (ROWPAIR_TRIATTN=flash); XLA_CLIENT_MEM_FRACTION=0.97 above 8,576
#           tokens (the 24-mer, 9,240 residues, folds at 0.97 and not at 0.95)
#   stock1  the validated stock configuration on one GCD; stock8 the stock row-sharded one on 8 GCDs (for comparison)
# Launch serialisation (fault (c), AF3JAX_PORT.md): --serialise copy, the default since N2 (job 50189815), serialises only the memory
#   copies (AMD_SERIALIZE_COPY=3): no crash in 12 seeds on one GCD and on 8, 4-9% faster on warm seeds, the same structures.
#   --serialise block (HIP_LAUNCH_BLOCKING=1, every launch) is the fallback, and what every run before N2 used.
# Compilation cache: compiled programs and XLA's autotuning results are kept in $AF3JAX_W/cache by default, so each bucket compiles once
#   (Q1, job 50229895: a warm cache cut the 12-mer's first seed on 8 GCDs from 809.7 to 600.8 s, output bit-identical; ~7 MB per
#   bucket). --cache_dir DIR (or AF3JAX_CACHE) puts it elsewhere; --cache_dir private compiles into a private directory deleted at exit
#   (one GCD and, since the A1 audit, 8 GCDs too: before, an 8-GCD run fell back to the runner's node-local /tmp/alphafold_cache,
#   shared by every run on the node, which let V1's setup-C run reuse the plain run's GEMM tuning). If the default directory
#   cannot be written (another user's tree), the run falls back to private with a note.
# GEMM backend (--gemm or AF3JAX_GEMM): triton = --xla_gpu_cublas_fallback=false, XLA's Triton GEMMs only (the validated one-GCD
#   default: crash (a), hipBLASLt's gfx90a tables name kernels no code object contains); library = --xla_gpu_cublas_fallback=true (the
#   autotuner may pick hipBLASLt; what every 8-GCD run uses, as a sharded GEMM fusion has no Triton config); rocblas = library
#   with --xla_gpu_enable_cublaslt=false, meant as Triton or rocBLAS. G4 (job 50388001) showed this XLA has no rocBLAS path and
#   ignores that flag, so 'rocblas' runs hipBLASLt exactly like 'library' (the name stays, for compatibility). In what follows,
#   'rocblas' means library (hipBLASLt) GEMMs. N3a (job 50247600): the library makes the flash
#   pairformer layer 1.38-1.49x faster. On 8 GCDs 'triton' is not available and means 'library'.
#   Default (no --gemm): rocblas for setup A up to bucket $AF3JAX_ROCBLAS_A_MAX (3,456) and setup B up to $AF3JAX_ROCBLAS_B_MAX (4,480,
#   B's top, since v13), triton for every other one-GCD run, library
#   on 8 GCDs. G2 (job 50273773, the official parameters, 4 seeds): rocblas 1.46-1.56x faster on warm seeds, inside triton's own
#   run-to-run noise; it compiles 2.5-13x longer (once per bucket with the cache) and needs 6-9% more memory: at 3,584 it does not fit
#   (triton does), so A from 3,457 to 3,584 keeps triton. Setup B: A1 (job 50350388) 1.20x on the 10-mer at 3,968, identical
#   confidence and per-chain accuracy, the same peak.
#   PAIR-LOGITS LEVER (v13): library GEMMs on one GCD are SILENTLY WRONG above about 4,314 tokens without it. G3 (job 50379642): rc 0
#   and finite confidences, but chains misfold. G6/G7 (jobs 50437098, 50456060): hipBLASLt's default algorithm computes the diffusion
#   transformer's pair_logits_projection (f32 [N^2, 128] x [128, 64], column-major result) wrongly from 18,612,224 rows, first bad
#   row 2^29 - 28 x rows (an AMD library fault). PERF_PAIRLOGITS_ROWS=R (perf_launch.py) computes that projection in R-token-row
#   chunks, values unchanged. F1 (job 50463106): with R = 1,024 or 2,048 every chain folds at 4,352 and 4,480 (0.42-0.53 A from
#   triton, pTM within 0.006, the same peaks, 1.09-1.10x cold). So any one-GCD run with library GEMMs above bucket
#   $AF3JAX_PAIRLOGITS_MIN (3,968) gets PERF_PAIRLOGITS_ROWS=1024 unless you set it (0 = off). Library GEMMs on one GCD above 4,224
#   with the lever off print a WARNING. 8 GCDs are not affected (rowpair shards the pair; A1's 12-mer at 4,736: 1.6 A on every chain).
# Triangle attention, rows per call: setup A up to bucket $AF3JAX_ATTN_CHUNK_MAX (2,560) runs it as one call over every row
#   (PERF_ATTN_CHUNK=none) instead of the fork's 32-row chunks above 1,536 tokens. AC1 (job 50324129): 1.17x on the 4-mer's warm
#   seed, 1.03x at 2,432, no memory cost, values within noise. A PERF_ATTN_CHUNK you set yourself wins; AF3JAX_ATTN_CHUNK_MAX=0 turns
#   the default off.
# AF3JAX_COMMAND_BUFFER=xla leaves XLA's command buffers on: it segfaults on this stack (crash (b) in AF3JAX_PORT.md; Q1), so every
#   validated run turns them off.
# auto: tokens are estimated from the input JSON (protein/RNA/DNA residues x copies; each ligand counted as 64 tokens, with a note),
# rounded up to a multiple of 128 -- the single explicit bucket every validation used (--bucket overrides; AlphaFold 3's own default list
# would pad a 1,540-token input to 2,048).
# Environment: AF3JAX_W (the tree; default: the directory above this script's scripts/, so $AF3JAX_W/scripts/af3jax_predict.sh
# finds its own tree, natively and in the image at /opt/af3jax), AF3JAX_A_MAX, AF3JAX_B_MAX, AF3JAX_CACHE,
# AF3JAX_COMMAND_BUFFER, AF3JAX_GEMM, AF3JAX_MODEL_DIR, AF3JAX_ATTN_CHUNK_MAX, PERF_ATTN_CHUNK, AF3JAX_ROCBLAS_A_MAX,
# AF3JAX_ROCBLAS_B_MAX, AF3JAX_PAIRLOGITS_MIN, PERF_PAIRLOGITS_ROWS. The alphafold3 repository and
# tokamax are copied to node-local $TMPDIR and patched there (patch_rocm_capability.py, patch_tokamax_rocm.py): never the shared tree.
set -uo pipefail
TAG="[af3jax-predict]"
W=${AF3JAX_W:-$(cd "$(dirname "$0")/.." && pwd)}
A_MAX=${AF3JAX_A_MAX:-3584}
B_MAX=${AF3JAX_B_MAX:-4480}
IN="" OUTD="" REGIME=auto GCD=0 BUCKET="" RECYCLES="" SERIALISE=copy DRY=0 EXTRA=() CACHE=${AF3JAX_CACHE:-}
CB=${AF3JAX_COMMAND_BUFFER:-off} GEMM=${AF3JAX_GEMM:-} MODEL_DIR=${AF3JAX_MODEL_DIR:-}
ROCBLAS_A_MAX=${AF3JAX_ROCBLAS_A_MAX:-3456}
ROCBLAS_B_MAX=${AF3JAX_ROCBLAS_B_MAX:-4480}
PAIRLOGITS_MIN=${AF3JAX_PAIRLOGITS_MIN:-3968}
ATTN_CHUNK_MAX=${AF3JAX_ATTN_CHUNK_MAX:-2560}
while [ $# -gt 0 ]; do
  case "$1" in
    --input_dir) IN=$2; shift 2;;
    --output_dir) OUTD=$2; shift 2;;
    --regime) REGIME=$2; shift 2;;
    --gcd) GCD=$2; shift 2;;
    --bucket) BUCKET=$2; shift 2;;
    --num_recycles) RECYCLES=$2; shift 2;;
    --serialise) SERIALISE=$2; shift 2;;
    --gemm) GEMM=$2; shift 2;;
    --model_dir) MODEL_DIR=$2; shift 2;;
    --cache_dir) CACHE=$2; shift 2;;
    --dry_run) DRY=1; shift;;
    -h|--help) sed -n '2,/^set -uo pipefail/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0;;
    --) shift; EXTRA=("$@"); break;;
    *) echo "$TAG unknown argument: $1"; exit 2;;
  esac
done
CACHE_DEFAULT=0
case "${CACHE:-default}" in default) CACHE=$W/cache; CACHE_DEFAULT=1;; private) CACHE="";; esac
[ -n "$IN" ] && [ -n "$OUTD" ] || { echo "usage: $0 --input_dir DIR --output_dir DIR [--regime auto|A|B|C|stock1|stock8] [--gcd N] [--bucket N] [--num_recycles R] [--serialise copy|block] [--cache_dir DIR] [--dry_run] [-- flags]"; exit 2; }
case "$REGIME" in auto|A|B|C|stock1|stock8) ;; *) echo "$TAG --regime $REGIME: auto, A, B, C, stock1 or stock8"; exit 2;; esac
case "$SERIALISE" in copy|block) ;; *) echo "$TAG --serialise $SERIALISE: copy or block"; exit 2;; esac
case "$CB" in off|xla) ;; *) echo "$TAG AF3JAX_COMMAND_BUFFER=$CB: off or xla"; exit 2;; esac
case "${GEMM:-triton}" in triton|library|rocblas) ;; *) echo "$TAG --gemm $GEMM: triton, library or rocblas"; exit 2;; esac
PY=$W/stockenv/bin/python
SITE=$W/stockenv/lib/python3.12/site-packages
if [ -n "$MODEL_DIR" ]; then WEIGHTS_FILE=$MODEL_DIR/af3.bin.zst; WEIGHTS="af3($MODEL_DIR)"; else WEIGHTS_FILE=$W/weights/p2/of3_ported_weights.bin.zst; WEIGHTS=of3-preview2; fi
for f in "$PY" "$W/alphafold/run_alphafold.py" "$WEIGHTS_FILE" "$W/scripts/perf_launch.py" "$W/scripts/rowpair_launch.py" \
         "$W/scripts/patch_rocm_capability.py" "$W/scripts/patch_tokamax_rocm.py"; do
  [ -e "$f" ] || { echo "$TAG MISSING: $f (a native tree on /scratch may have been purged; in the image, bind the weights or pass --model_dir)"; exit 3; }
done
n_json=$(ls "$IN"/*.json 2>/dev/null | wc -l)
[ "$n_json" -ge 1 ] || { echo "$TAG no *.json in $IN"; exit 2; }
[ "$n_json" -eq 1 ] || echo "$TAG note: $n_json inputs in $IN; the bucket and regime are chosen for the largest"

# ---- size: tokens from the JSON, the bucket, the regime
read -r TOKENS NOTE < <("$PY" - "$IN" <<'PYEOF'
import glob, json, os, sys
best, notes = 0, []
for p in glob.glob(os.path.join(sys.argv[1], "*.json")):
    d = json.load(open(p))
    t = 0
    for e in d.get("sequences", []):
        (kind, v), = e.items()
        ids = v.get("id", [])
        n = len(ids) if isinstance(ids, list) else 1
        if kind in ("protein", "rna", "dna"):
            t += len(v.get("sequence", "")) * n
        else:
            t += 64 * n
            notes.append(f"{kind}x{n}~64")
    best = max(best, t)
print(best, ",".join(notes) or "-")
PYEOF
)
[[ "${TOKENS:-}" =~ ^[0-9]+$ ]] && [ "$TOKENS" -gt 0 ] || { echo "$TAG could not read a token count from $IN/*.json"; exit 2; }
[ -n "$BUCKET" ] || BUCKET=$(( (TOKENS + 127) / 128 * 128 ))
if [ "$REGIME" = auto ]; then
  if [ "$BUCKET" -le "$A_MAX" ]; then REGIME=A; elif [ "$BUCKET" -le "$B_MAX" ]; then REGIME=B; else REGIME=C; fi
fi

# ---- the configuration of the chosen regime
FRACTION=0.95
case "$REGIME" in
  A)      LAUNCH=one;  LEVERS="cond_share,hoist,hoist_logits,sampler_bf16,triattn_flash"; MEM="" ;;
  B)      LAUNCH=one;  LEVERS="sampler_bf16,triattn_flash"; MEM="cond_shard,samples_per_pass" ;;
  stock1) LAUNCH=one;  LEVERS=""; MEM="" ;;
  C)      LAUNCH=many; LEVERS="flash"; MEM=""; [ "$BUCKET" -gt 8576 ] && FRACTION=0.97 ;;
  stock8) LAUNCH=many; LEVERS=""; MEM="" ;;
esac
ATTN_SRC=fork
if [ -n "${PERF_ATTN_CHUNK:-}" ]; then ATTN_SRC=given
elif [ "$REGIME" = A ] && [ "$BUCKET" -le "$ATTN_CHUNK_MAX" ]; then export PERF_ATTN_CHUNK=none; ATTN_SRC=auto; fi
GEMM_SRC=given; [ -n "$GEMM" ] || GEMM_SRC=auto
if [ "$LAUNCH" = one ]; then
  if [ -z "$GEMM" ]; then
    if { [ "$REGIME" = A ] && [ "$BUCKET" -le "$ROCBLAS_A_MAX" ]; } || { [ "$REGIME" = B ] && [ "$BUCKET" -le "$ROCBLAS_B_MAX" ]; }; then
      GEMM=rocblas; else GEMM=triton; fi
  fi
else
  [ "${GEMM:-library}" = triton ] && echo "$TAG note: --gemm triton is not available on 8 GCDs (a sharded GEMM fusion has no Triton config): library"
  [ "${GEMM:-library}" = rocblas ] || GEMM=library
fi
PL_SRC=off
if [ -n "${PERF_PAIRLOGITS_ROWS:-}" ]; then
  if [ "$PERF_PAIRLOGITS_ROWS" -gt 0 ] 2>/dev/null; then PL_SRC=given; else unset PERF_PAIRLOGITS_ROWS; PL_SRC=zero; fi
elif [ "$LAUNCH" = one ] && [ "$GEMM" != triton ] && [ "$BUCKET" -gt "$PAIRLOGITS_MIN" ]; then
  export PERF_PAIRLOGITS_ROWS=1024; PL_SRC=auto            # G6/G7/F1: the diffusion pair-logits projection in row chunks
fi
[ "$LAUNCH" = one ] || { unset PERF_PAIRLOGITS_ROWS; [ "$PL_SRC" = given ] && PL_SRC=8gcd; }
case "$GEMM" in triton) GEMMFLAGS="--xla_gpu_cublas_fallback=false";; library) GEMMFLAGS="--xla_gpu_cublas_fallback=true";;
  rocblas) GEMMFLAGS="--xla_gpu_cublas_fallback=true --xla_gpu_enable_cublaslt=false";; esac
echo "$TAG tokens~$TOKENS (ligand estimate: $NOTE) bucket=$BUCKET regime=$REGIME launcher=$([ $LAUNCH = one ] && echo "perf_launch.py GCD $GCD" || echo "rowpair_launch.py 8 GCDs") levers=${LEVERS:-none} mem_levers=${MEM:-none} XLA_CLIENT_MEM_FRACTION=$FRACTION serialise=$SERIALISE recycles=${RECYCLES:-default} cache=${CACHE:-private}$([ $CACHE_DEFAULT = 1 ] && echo "(default)") command_buffer=$CB gemm=$GEMM($GEMM_SRC) attn_chunk=${PERF_ATTN_CHUNK:-fork}($ATTN_SRC) pairlogits_rows=${PERF_PAIRLOGITS_ROWS:-off}($PL_SRC) weights=$WEIGHTS"
[ "$LAUNCH" = one ] && [ "$GEMM" != triton ] && [ "$BUCKET" -gt 4224 ] && [ -z "${PERF_PAIRLOGITS_ROWS:-}" ] && echo "$TAG WARNING: gemm=$GEMM on one GCD at bucket $BUCKET with the pair-logits lever off: library (hipBLASLt) GEMMs give silently wrong structures above about 4,314 tokens (G3/G6/G7: chains misfold, rc 0). Leave PERF_PAIRLOGITS_ROWS unset (the wrapper sets 1024) or use --gemm triton"

# ---- environment (the validated settings, AF3JAX_PORT.md "Go / no-go")
type module >/dev/null 2>&1 || { [ -n "${LMOD_PKG:-}" ] && source "$LMOD_PKG/init/bash" >/dev/null 2>&1; }
module use /software/setonix/unsupported >/dev/null 2>&1
module load rocm/7.2.4                   >/dev/null 2>&1
export ROCM_PATH=${ROCM_PATH:-/software/setonix/rocm/rocm-7.2.4}
export LD_LIBRARY_PATH=$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}
export TMPDIR=${TMPDIR:-/tmp}/af3jax-predict.$$ XDG_CACHE_HOME=${TMPDIR:-/tmp}/af3jax-predict.$$/xdg
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
trap 'rm -rf "$TMPDIR"' EXIT
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=$FRACTION JAX_PALLAS_USE_MOSAIC_GPU=0
unset XLA_PYTHON_CLIENT_MEM_FRACTION AMD_SERIALIZE_KERNEL AMD_SERIALIZE_COPY HIP_LAUNCH_BLOCKING HSA_ENABLE_SDMA JAX_COMPILATION_CACHE_DIR
unset TOKAMAX_ROCM_TRITON HIP_VISIBLE_DEVICES                  # gated linear units on XLA; flash is enabled for attention alone
if [ "$SERIALISE" = block ]; then export HIP_LAUNCH_BLOCKING=1; else export AMD_SERIALIZE_COPY=3; fi
export AF3JAX_KIT=$W/ubm/af3_jax/opt/af3_jax_opt AF3JAX_ADDON=$W/ubm/af3_jax/opt/forward/flashpairformer AF3JAX_CORE=$W/ubm/common/opt_core
BASE="--xla_disable_hlo_passes=triton-softmax-rewriter"
[ "$CB" = off ] && BASE="--xla_gpu_enable_command_buffer= $BASE"   # crash (b): HIP graph updates fail in the module's runtime
if [ -n "$MODEL_DIR" ]; then WFLAGS=(--model_dir="$MODEL_DIR"); else WFLAGS=(--of3_weights --model_dir="$W/weights/p2"); fi
RFLAGS=(--norun_data_pipeline --input_dir="$IN" --output_dir="$OUTD" "${WFLAGS[@]}" --buckets="$BUCKET"
        --flash_attention_implementation=xla)
[ -n "$RECYCLES" ] && RFLAGS+=(--num_recycles="$RECYCLES")
if [ -n "$CACHE" ] && [ "$DRY" != 1 ] && ! { mkdir -p "$CACHE" 2>/dev/null && [ -w "$CACHE" ]; }; then
  if [ "$CACHE_DEFAULT" = 1 ]; then echo "$TAG note: the default cache $CACHE is not writable; compiling privately"; CACHE=""
  else echo "$TAG cannot create or write --cache_dir $CACHE"; exit 2; fi
fi
if [ "$LAUNCH" = one ]; then
  export ROCR_VISIBLE_DEVICES=$GCD XLA_FLAGS="$BASE $GEMMFLAGS" PERF_LEVERS="$LEVERS" PERF_MEM_LEVERS="$MEM"
  CMD=("$PY" "$W/scripts/perf_launch.py" "$TMPDIR/alphafold/run_alphafold.py" "${RFLAGS[@]}" --cache_dir="${CACHE:-$TMPDIR/cache}" ${EXTRA[@]+"${EXTRA[@]}"})
else
  unset ROCR_VISIBLE_DEVICES
  export XLA_FLAGS="$BASE $GEMMFLAGS --xla_gpu_unsupported_use_all_reduce_one_shot_kernel=false"
  export AF3_JAX_N_GPU=8 ROWPAIR_SCHEDULE=ring ROWPAIR_TRIATTN=$([ "$REGIME" = C ] && echo flash || echo "")
  CMD=("$PY" "$W/scripts/rowpair_launch.py" "$TMPDIR/alphafold/run_alphafold.py" "${RFLAGS[@]}" --cache_dir="${CACHE:-$TMPDIR/cache}" ${EXTRA[@]+"${EXTRA[@]}"})
fi
if [ "$DRY" = 1 ]; then
  echo "$TAG dry run: XLA_FLAGS='$XLA_FLAGS' ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-all} PERF_LEVERS=${PERF_LEVERS:-} PERF_MEM_LEVERS=${PERF_MEM_LEVERS:-} PERF_PAIRLOGITS_ROWS=${PERF_PAIRLOGITS_ROWS:-} ROWPAIR_TRIATTN=${ROWPAIR_TRIATTN:-} $([ "$SERIALISE" = block ] && echo HIP_LAUNCH_BLOCKING=1 || echo AMD_SERIALIZE_COPY=3)"
  echo "$TAG dry run: ${CMD[*]}"
  exit 0
fi

# ---- private, node-local patched copies (never the shared tree)
cp -a "$W/alphafold" "$TMPDIR/alphafold"
[ -f "$W/alphafold/run_alphafold.py.rocm-orig" ] && cp "$W/alphafold/run_alphafold.py.rocm-orig" "$TMPDIR/alphafold/run_alphafold.py"
mkdir -p "$TMPDIR/pylib"; cp -a "$SITE/tokamax" "$TMPDIR/pylib/tokamax"
for f in gpu_utils.py precision.py; do
  [ -f "$SITE/tokamax/_src/$f.rocm-orig" ] && cp "$SITE/tokamax/_src/$f.rocm-orig" "$TMPDIR/pylib/tokamax/_src/$f"
done
"$PY" "$W/scripts/patch_rocm_capability.py" "$TMPDIR/alphafold" >/dev/null && "$PY" "$W/scripts/patch_tokamax_rocm.py" "$TMPDIR/pylib" >/dev/null \
  || { echo "$TAG patching the private copies failed"; exit 3; }
export PYTHONPATH=$TMPDIR/pylib${PYTHONPATH:+:$PYTHONPATH}
"${CMD[@]}"
