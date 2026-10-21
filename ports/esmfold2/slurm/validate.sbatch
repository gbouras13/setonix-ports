#!/bin/bash
#SBATCH --job-name=esmfold2-validate
#SBATCH --partition=gpu-highmem
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=esmfold2-validate.%j.out

# The esmfold2 image against the native port, on one Setonix GPU node (8 MI250X GCDs).
#   sbatch --account=<project>-gpu validate.sbatch SIF=<the .sif> OUT=<dir> NATIVE_D=<the native esmfold2-setonix tree> \
#          [REF=<native outputs, default $NATIVE_D/out/valid6.51248511>] [WEIGHTS=<default $NATIVE_D/weights>] [PARTS=check,bytes,fetch]
#   (arguments, not variables: Setonix sets SBATCH_EXPORT=NONE). PARTS picks the sections, default all three.
#
# 1. check  `esmfold2 check --predict`: versions, the kit's pin check, the weights through the kit's own sha256 check, every GPU
#           library loaded from inside the image, a matmul on each of the 8 GCDs, the nvidia-smi shim, and 1BRS (199 tokens) through
#           stock-fused and exact inside the image, compared byte for byte.
# 2. bytes  the 48 bench targets (105-497 residues; the 11 MSA inputs for full_msa), seed 0, --det 1, library defaults, every route
#           the native job valid6 (51248511) ran through scripts/ef2.sh, one GCD each, side by side as natively: exact and fast on
#           ESMFold2-Fast and on ESMFold2 without MSA, exact with MSA, stock-fused on both models, big. Each against its native
#           output, byte for byte. This is the question the image has to answer: an interpreter built elsewhere
#           (python-build-standalone 3.12.12, not cray-python 3.12.12), a venv built elsewhere from the same lock, and torch's ROCm
#           kernels pruned to gfx90a must still produce the native bytes on every route.
# 3. fetch  `esmfold2 fetch-weights` into node-local disk: the 16 files from Hugging Face at their pinned commits, the kit's own sha256
#           check, then a byte comparison with the native copies. This is the image's answer to the /scratch purge.
set -uo pipefail
for a in "$@"; do
  case "$a" in
    SIF=*|OUT=*|NATIVE_D=*|REF=*|WEIGHTS=*|PARTS=*) export "${a?}" ;;
    *) echo "unknown argument '$a': SIF= OUT= NATIVE_D= [REF=] [WEIGHTS=] [PARTS=]"; exit 2 ;;
  esac
done
PARTS=${PARTS:-check,bytes,fetch}
has() { case ",$PARTS," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
: "${SIF:?SIF=}" "${OUT:?OUT=}" "${NATIVE_D:?NATIVE_D=}"
REF=${REF:-$NATIVE_D/out/valid6.51248511}
WEIGHTS=${WEIGHTS:-$NATIVE_D/weights}
BENCH=$NATIVE_D/inputs/bench/ef2bench48.json
MSA=$NATIVE_D/inputs/msa11.json
O=$OUT/validate.${SLURM_JOB_ID:-local}
mkdir -p "$O" && cd "$O" || exit 3
for f in "$SIF" "$WEIGHTS/hub" "$BENCH" "$MSA" "$REF/exact/cif_all"; do
  [ -e "$f" ] || { echo "MISSING: $f"; exit 3; }
done
module load singularity/4.1.0-nohost > /dev/null 2>&1 || { echo "ABORT: no singularity/4.1.0-nohost module"; exit 3; }
L=/tmp/efs-validate-$USER-${SLURM_JOB_ID:-$$}
mkdir -p "$L"
export SINGULARITY_CACHEDIR=$L/sing-cache APPTAINER_CACHEDIR=$L/sing-cache
echo "node-local /tmp: $(df -h --output=avail /tmp | tail -1 | tr -d ' ') free"
# The image's writable places, all node-local and per job: the staged weights (one copy for every lane), compiled kernels, temporaries.
# The kit's weights check writes refs/main and a digest memo: the memo goes under EFS_CACHE, not into the native tree.
export EFS_WEIGHTS_LOCAL=$L/weights-staged EFS_CACHE=$L/cache
trap 'rm -rf "$L"' EXIT
C=(singularity exec --bind "$L:$L" --bind "$O:$O" "$SIF" esmfold2)    # never --rocm
FAIL=0
note() { printf '\n######## %s\n' "$*"; }

echo "=== node $(hostname)  job ${SLURM_JOB_ID:-?}  $(date)  image $SIF ($(du -h "$SIF" | cut -f1)) ==="
"${C[@]}" version | sed 's/^/  /'
t0=$SECONDS; cat "$SIF" > /dev/null; echo "pre-read the image: $((SECONDS - t0)) s"

# ---------------------------------------------------------------------------------------------------- 3 (started first: network only)
# 27.5 GB more on the node's /tmp beside the staged copy; on /scratch (removed at the end) when /tmp has under 100 GB free
FW=$L/weights-fetched
[ "$(df -BG --output=avail /tmp | tail -1 | tr -dc 0-9)" -ge 100 ] || FW=$O/weights-fetched
if has fetch; then
  ( EFS_LOCAL=$L/fetch-local "${C[@]}" fetch-weights "$FW" > "$O/fetch.log" 2>&1; echo "rc=$?" >> "$O/fetch.log" ) &
  FETCH_PID=$!
fi

# one staged copy of the native weights, before any lane starts
"${C[@]}" stage --weights "$WEIGHTS" > "$O/stage.log" 2>&1 || { echo "ABORT: staging failed"; cat "$O/stage.log"; exit 3; }
tail -2 "$O/stage.log" | sed 's/^/  /'

# ---------------------------------------------------------------------------------------------------- 1. check
if has check; then
note "1. esmfold2 check --predict (all 8 GCDs; 1BRS, 199 tokens, stock-fused vs exact inside the image)"
EFS_LOCAL=$L/check "${C[@]}" check --weights "$WEIGHTS" --predict --keep "$O/check_predict" > check.log 2>&1; rc=$?
sed 's/^/  /' check.log
echo "check rc=$rc"; [ "$rc" -eq 0 ] || FAIL=1
fi

# ---------------------------------------------------------------------------------------------------- 2. bytes
if has bytes; then
note "2. the bench set, seed 0, --det 1: every route in the image against the native job valid6 (51248511)"
# tag (= the native lane)  GCD  route  variant  input
RUNS=("exact|0|exact|fast|$BENCH"
      "fast_notx|1|fast|fast|$BENCH"
      "F_exact|2|exact|full_nomsa|$BENCH"
      "F_fast_notx|3|fast|full_nomsa|$BENCH"
      "off_fused|4|stock-fused|fast|$BENCH"
      "F_off_fused|5|stock-fused|full_nomsa|$BENCH"
      "M_exact|6|exact|full_msa|$MSA"
      "big_allocoff|7|big|fast|$BENCH")
PIDS=()
for r in "${RUNS[@]}"; do
  IFS='|' read -r tag gcd route variant input <<< "$r"
  d=$O/bytes/$tag; mkdir -p "$d"
  (
    export ROCR_VISIBLE_DEVICES=$gcd EFS_LOCAL=$L/run-$tag
    t0=$SECONDS
    "${C[@]}" pred "$route" --weights "$WEIGHTS" --variant "$variant" --input "$input" --out_dir "$d/out" --seeds 0 --det 1 > "$d/run.log" 2>&1
    echo "RESULT tag=$tag gcd=$gcd route=$route variant=$variant rc=$? wall_s=$((SECONDS - t0))" > "$d/result.txt"
  ) &
  PIDS+=($!)
  sleep 20
done
wait "${PIDS[@]}"
for r in "${RUNS[@]}"; do
  IFS='|' read -r tag _ <<< "$r"
  res=$(cat "$O/bytes/$tag/result.txt" 2>/dev/null || echo "RESULT tag=$tag (none)")
  if [ -d "$REF/$tag/cif_all" ] && [ -d "$O/bytes/$tag/out/cif_all" ]; then
    n_ref=$(find "$REF/$tag/cif_all" -name '*.cif' | wc -l); n_img=$(find "$O/bytes/$tag/out/cif_all" -name '*.cif' | wc -l)
    cmp=$("${C[@]}" compare "$REF/$tag" "$O/bytes/$tag/out" --quiet 2>&1 | tail -1); crc=$?
    if [ "$crc" -eq 0 ] && [ "$n_ref" = "$n_img" ]; then v=IDENTICAL; else v=DIFFER; FAIL=1; fi
    cmp="$n_img/$n_ref structures; $cmp"
  else
    v=MISSING; cmp="(no output, or no native reference at $REF/$tag)"; FAIL=1
  fi
  case "$res" in *" rc=0 "*) ;; *) FAIL=1 ;; esac
  printf '  %-75s %-9s %s\n' "$res" "$v" "$(echo "$cmp" | cut -c1-160)"
done
fi

# ---------------------------------------------------------------------------------------------------- 3. fetch (collect)
if has fetch; then
note "3. esmfold2 fetch-weights (Hugging Face at the pinned commits, the kit's sha256 check), then against the native copies"
wait "$FETCH_PID"
grep -a -E '^\[fetch\] (curl|OK|FAILED|SIZE)|WEIGHTS|^rc=' "$O/fetch.log" | sed 's/^/  /'
grep -q '^rc=0$' "$O/fetch.log" || FAIL=1
nd=0; nf=0
while IFS= read -r f; do
  nf=$((nf + 1))
  cmp -s "$FW/$f" "$WEIGHTS/$f" || { echo "  $f DIFFERS from the native copy (or is missing)"; nd=$((nd + 1)); FAIL=1; }
done < <(cd "$WEIGHTS" && find hub -path '*/snapshots/*' -type f)
echo "  $((nf - nd))/$nf weight files byte-identical to the native copies (fetched into $FW)"
[ "$FW" = "$L/weights-fetched" ] || rm -rf "$FW"
fi

note "VALIDATE $([ "$FAIL" -eq 0 ] && echo PASS || echo FAILED)  (job ${SLURM_JOB_ID:-local}, $(date +%T))"
echo "outputs in $O"
exit "$FAIL"
