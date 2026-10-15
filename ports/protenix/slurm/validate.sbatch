#!/bin/bash
#SBATCH --job-name=protenix-validate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=protenix-validate.%j.out

# The protenix image against the native port, on one Setonix GPU node (8 MI250X GCDs).
#   sbatch --account=<project>-gpu validate.sbatch SIF=<the .sif> OUT=<dir> NATIVE_D=<the native protenix-setonix tree> \
#          [WEIGHTS=<default $OUT/weights>] [REF1=<default $NATIVE_D/out/vdet.50030425>] \
#          [REF8=<default $NATIVE_D/out/spd2.50379609/n2048_p8_flashcomm>] [PARTS=fetch,check,bytes]
#   (arguments, not variables: Setonix sets SBATCH_EXPORT=NONE). PARTS picks the sections, default all three.
#
# The native tree is only read: its stored outputs, and what is left of its weights. The /scratch purge took four of its six data caches
# and most of its venv on 2026-10-14, so the image fetches its own weights first, into WEIGHTS, and every section runs on those.
# 1. fetch  `protenix fetch-weights WEIGHTS`: the checkpoint from the mirror, the six caches through the kit's own routine, all seven
#           checked against the kit's pins (files already there are kept and checked). Each is then compared with the native copy where
#           one survives. This is the image's answer to the purge.
# 2. check  `protenix check --predict`: versions, the kit's pin check, the seven weight files through the kit's own check, every GPU
#           library loaded from inside the image, a matmul on each of the 8 GCDs, and the kit's 1BRS example (199 tokens, stock defaults:
#           10 cycles, 200 steps, 5 samples, seed 101) through stock's own CLI and through the one-GCD big line inside the image, under
#           deterministic algorithms, compared byte for byte.
# 3. bytes  the image against the native port's own deterministic runs, byte for byte. This is the question the image has to answer: an
#           interpreter built elsewhere (python-build-standalone 3.12.12, not the OpenFold3 env's 3.12.14), a flat venv instead of the
#           layered one, and torch's ROCm kernels pruned to gfx90a must still produce the native bytes on every line:
#             S1  check's stock arm      against the native stock CLI run S1 of job 50030425
#             K1  check's big arm        against the native one-GCD big run K1 of job 50030425 (identical to S1 there)
#             T8  big --n_gpu 8 on all 8 GCDs: the native 2,048-token query, 1 cycle, 10 steps, 5 samples, seed 101, deterministic, with
#                 the current defaults (the engine's DiT attention, opt_core's fused triangle attention, the RCCL settings, chunk cap 4),
#                 against the native run n2048_p8_flashcomm of job 50379609 (whose repeat, n2048_p8_flashcomm2, was identical to it)
set -uo pipefail
for a in "$@"; do
  case "$a" in
    SIF=*|OUT=*|NATIVE_D=*|REF1=*|REF8=*|WEIGHTS=*|PARTS=*) export "${a?}" ;;
    *) echo "unknown argument '$a': SIF= OUT= NATIVE_D= [WEIGHTS=] [REF1=] [REF8=] [PARTS=]"; exit 2 ;;
  esac
done
PARTS=${PARTS:-fetch,check,bytes}
has() { case ",$PARTS," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
: "${SIF:?SIF=}" "${OUT:?OUT=}" "${NATIVE_D:?NATIVE_D=}"
WEIGHTS=${WEIGHTS:-$OUT/weights}
REF1=${REF1:-$NATIVE_D/out/vdet.50030425}
REF8=${REF8:-$NATIVE_D/out/spd2.50379609/n2048_p8_flashcomm}
O=$OUT/validate.${SLURM_JOB_ID:-local}
mkdir -p "$O" "$WEIGHTS" && cd "$O" || exit 3
[ -e "$SIF" ] || { echo "MISSING: $SIF"; exit 3; }
module load singularity/4.1.0-nohost > /dev/null 2>&1 || { echo "ABORT: no singularity/4.1.0-nohost module"; exit 3; }
L=/tmp/pxs-validate-$USER-${SLURM_JOB_ID:-$$}
mkdir -p "$L"
export SINGULARITY_CACHEDIR=$L/sing-cache APPTAINER_CACHEDIR=$L/sing-cache
# The kit's cache root (its weights-digest memo): node-local, so this job writes nothing beside any weights but the ones it fetches.
export PROTENIX_OPT_CACHE_DIR=$L/kit-cache
C=(singularity exec --bind "$L:$L" --bind "$O:$O" --bind "$WEIGHTS:$WEIGHTS" "$SIF" protenix)    # never --rocm
FILES="checkpoint/protenix-v2.pt common/components.cif common/components.cif.rdkit_mol.pkl common/obsolete_release_date.csv
       common/clusters-by-entity-40.txt common/release_date_cache.json common/obsolete_to_successor.json"
FAIL=0
note() { printf '\n######## %s\n' "$*"; }
same() {   # same <tag> <native dir> <image dir>: every sample's CIF and confidence JSON byte-identical
  local tag=$1 ref=$2 mine=$3 out rc v
  if [ ! -d "$ref" ] || [ ! -d "$mine" ]; then
    printf '  %-3s MISSING  (native %s: %s; image %s: %s)\n' "$tag" "$ref" "$([ -d "$ref" ] && echo ok || echo absent)" \
      "$mine" "$([ -d "$mine" ] && echo ok || echo absent)"
    FAIL=1; return
  fi
  out=$("${C[@]}" compare --identical "$ref" "$mine" 2>&1); rc=$?
  if [ "$rc" -eq 0 ]; then v=IDENTICAL; else v=DIFFER; FAIL=1; fi
  printf '  %-3s %-9s %s\n' "$tag" "$v" "$(printf '%s\n' "$out" | grep -a 'byte-identical' | tail -1 | sed -E 's/^ *//')"
  printf '%s\n' "$out" | grep -a 'bytes_identical=' | sed 's/^ */      /' | cut -c1-200
}

echo "=== node $(hostname)  job ${SLURM_JOB_ID:-?}  $(date)  image $SIF ($(du -h "$SIF" | cut -f1)) ==="
"${C[@]}" version | sed 's/^/  /'
t0=$SECONDS; cat "$SIF" > /dev/null; echo "pre-read the image: $((SECONDS - t0)) s"

# ---------------------------------------------------------------------------------------------------- 1. fetch
if has fetch; then
note "1. protenix fetch-weights $WEIGHTS (the kit's pins), then against the native copies that survive"
t0=$SECONDS
PXS_LOCAL=$L/fetch "${C[@]}" fetch-weights "$WEIGHTS" > "$O/fetch.log" 2>&1; rc=$?
grep -a -E '^\[fetch\]|WEIGHTS OK|FAILED|REFUSED|absent|present' "$O/fetch.log" | sed 's/^/  /' | cut -c1-220
echo "fetch rc=$rc wall_s=$((SECONDS - t0))"; [ "$rc" -eq 0 ] || FAIL=1
for f in $FILES; do
  if [ ! -e "$NATIVE_D/weights/$f" ]; then echo "  $f: no native copy left (the /scratch purge)"
  elif cmp -s "$WEIGHTS/$f" "$NATIVE_D/weights/$f"; then echo "  $f = native"
  else echo "  $f DIFFERS from the native copy (or is missing here)"; FAIL=1; fi
done
fi
for f in $FILES; do
  [ -s "$WEIGHTS/$f" ] || { note "VALIDATE FAILED: $WEIGHTS/$f is missing, so nothing can run (PARTS=fetch fetches it)"; exit 1; }
done

# ---------------------------------------------------------------------------------------------------- 2. check
if has check || has bytes; then
note "2. protenix check --predict (all 8 GCDs; 1BRS, 199 tokens, stock defaults: stock vs the one-GCD big line, deterministic)"
PXS_LOCAL=$L/check "${C[@]}" check --weights "$WEIGHTS" --predict --keep "$O/check" > check.log 2>&1; rc=$?
sed 's/^/  /' check.log
echo "check rc=$rc"; [ "$rc" -eq 0 ] || FAIL=1
fi

# ---------------------------------------------------------------------------------------------------- 3. bytes
if has bytes; then
note "3. the image against the native port's deterministic runs, byte for byte"
same S1 "$REF1/S1/out" "$O/check/stock/out"
same K1 "$REF1/K1/out" "$O/check/big/out"
if [ -s "$REF8/q.json" ]; then
  d=$O/bytes/T8; mkdir -p "$d"
  t0=$SECONDS
  PXS_LOCAL=$L/t8 PXS_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    "${C[@]}" pred --weights "$WEIGHTS" --n_gpu 8 -i "$REF8/q.json" -o "$d/out" \
      --use_msa false --seeds 101 --cycle 1 --step 10 --sample 5 > "$d/run.log" 2>&1
  rc=$?
  echo "  T8  rc=$rc wall_s=$((SECONDS - t0))   native: $(head -1 "$REF8/verdict.txt" 2>/dev/null | cut -c1-200)"
  [ "$rc" -eq 0 ] || { FAIL=1; grep -a -E "Traceback|Error|NOT ACTIVE|\[ptx-pred\]" "$d/run.log" | tail -5 | sed 's/^/      /'; }
  grep -a -h -E "pxs-det r0\] exit|TP-ROUTE-LAUNCH|EXECUTION .*FINAL" "$d/run.log" 2>/dev/null | sort -u | cut -c1-200 | sed 's/^/      /'
  same T8 "$REF8/out" "$d/out"
  [ -d "${REF8}2/out" ] && same T8n "$REF8/out" "${REF8}2/out"    # the native run against its own repeat (job 50379609)
else
  echo "  T8  MISSING: no native query at $REF8/q.json"; FAIL=1
fi
fi

note "VALIDATE $([ "$FAIL" -eq 0 ] && echo PASS || echo FAILED)  (job ${SLURM_JOB_ID:-local}, $(date +%T))"
echo "outputs in $O; weights in $WEIGHTS"
exit "$FAIL"
