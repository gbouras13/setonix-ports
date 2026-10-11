#!/bin/bash
#SBATCH --job-name=boltz2-validate
#SBATCH --partition=gpu-highmem
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=boltz2-validate.%j.out

# The boltz2 image against the native port, on one Setonix GPU node (8 MI250X GCDs).
#   sbatch --account=<project>-gpu validate.sbatch SIF=<the .sif> OUT=<dir> NATIVE_D=<the native boltz2-setonix tree> \
#          [REF=<native outputs, default $NATIVE_D/out/fastdef.50579785>] [WEIGHTS=<default $NATIVE_D/weights>] [PARTS=check,bytes,fetch]
#   (arguments, not variables: Setonix sets SBATCH_EXPORT=NONE). PARTS picks the sections, default all three.
#
# 1. check  `boltz2 check --predict`: versions, the kit's pin check, the weights through the kit's own weights verb, every GPU library
#           loaded from inside the image, a matmul on each of the 8 GCDs, the nvidia-smi shim, and the kit's 398-token example through
#           stock and through exact inside the image, compared byte for byte (exact replays a captured graph: DEBUG_CLR_GRAPH_PACKET_
#           CAPTURE=0 must be in force).
# 2. bytes  the HK97 capsid trimer with its MSA (1,155 tokens), seed 1, upstream settings, in the image, laid out as the native job
#           50579785 ran it (one GCD each, the sharded run on two): stock, exact, fast, fast with the row patch, the one-GCD big default
#           and big --n_gpu 2. Each against its native output, byte for byte. This is the question the image has to answer: an
#           interpreter built elsewhere (python-build-standalone 3.12.12, not the OpenFold3 env's 3.12.14), a flat venv instead of the
#           layered one, and torch's ROCm kernels pruned to gfx90a must still produce the native bytes in every mode.
# 3. fetch  `boltz2 fetch-weights` into node-local disk: the four files from Hugging Face, each checked against the kit's pinned sha256
#           by the script, then compared with the native copies. This is the image's answer to the /scratch purge.
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
REF=${REF:-$NATIVE_D/out/fastdef.50579785}
WEIGHTS=${WEIGHTS:-$NATIVE_D/weights}
YH=$NATIVE_D/queries/hk97/hk97_trimer_msa.yaml
O=$OUT/validate.${SLURM_JOB_ID:-local}
mkdir -p "$O" && cd "$O" || exit 3
for f in "$SIF" "$WEIGHTS/boltz2_conf.ckpt" "$WEIGHTS/mols.tar" "$YH"; do
  [ -e "$f" ] || { echo "MISSING: $f"; exit 3; }
done
module load singularity/4.1.0-nohost > /dev/null 2>&1 || { echo "ABORT: no singularity/4.1.0-nohost module"; exit 3; }
L=/tmp/b2s-validate-$USER-${SLURM_JOB_ID:-$$}
mkdir -p "$L"
export SINGULARITY_CACHEDIR=$L/sing-cache APPTAINER_CACHEDIR=$L/sing-cache
# The digest memo of the kit's weights check: here, not beside the native weights (this job writes nothing into the native tree).
export MODEL_OPT_WEIGHTS_DIGEST_DIR=$L/digest
C=(singularity exec --bind "$L:$L" --bind "$O:$O" "$SIF" boltz2)    # never --rocm
FAIL=0
note() { printf '\n######## %s\n' "$*"; }

echo "=== node $(hostname)  job ${SLURM_JOB_ID:-?}  $(date)  image $SIF ($(du -h "$SIF" | cut -f1)) ==="
"${C[@]}" version | sed 's/^/  /'
t0=$SECONDS; cat "$SIF" > /dev/null; echo "pre-read the image: $((SECONDS - t0)) s"

# ---------------------------------------------------------------------------------------------------- 3 (started first: network only)
if has fetch; then
  ( B2S_LOCAL=$L/fetch "${C[@]}" fetch-weights "$L/weights" > "$O/fetch.log" 2>&1; echo "rc=$?" >> "$O/fetch.log" ) &
  FETCH_PID=$!
fi

# ---------------------------------------------------------------------------------------------------- 1. check
if has check; then
note "1. boltz2 check --predict (all 8 GCDs; 1BRS x2, 398 tokens, stock vs exact inside the image)"
B2S_LOCAL=$L/check "${C[@]}" check --weights "$WEIGHTS" --predict --keep "$O/check_predict" > check.log 2>&1; rc=$?
sed 's/^/  /' check.log
echo "check rc=$rc"; [ "$rc" -eq 0 ] || FAIL=1
fi

# ---------------------------------------------------------------------------------------------------- 2. bytes
if has bytes; then
note "2. HK97 trimer (1,155 tokens), seed 1, upstream settings: every mode in the image against the native job 50579785"
# tag  GCDs  extra environment  mode arguments          (as the native fastdef.sbatch ran them)
RUNS=("F1|0|B2S_FAST_PATCH=1|--mode fast"
      "F2|1||--mode fast"
      "E1|2||--mode exact"
      "S1|3||--mode off"
      "B1|4||--mode big"
      "T2|6,7||--mode big --n_gpu 2")
PIDS=()
for r in "${RUNS[@]}"; do
  IFS='|' read -r tag gcds extra margs <<< "$r"
  d=$O/bytes/$tag; mkdir -p "$d"
  (
    cd "$d" || exit 3
    export ROCR_VISIBLE_DEVICES=$gcds B2S_LOCAL=$L/run-$tag
    [ -n "$extra" ] && export "${extra?}"
    t0=$SECONDS
    # shellcheck disable=SC2086
    "${C[@]}" pred --weights "$WEIGHTS" $margs --input "$YH" --out_dir "$d/out" --seed 1 > "$d/run.log" 2>&1
    echo "RESULT tag=$tag gcds=$gcds rc=$? wall_s=$((SECONDS - t0)) $(grep -a -o -E 'total_s=[0-9.]+' "$d/run.log" | tail -1)" > "$d/result.txt"
  ) &
  PIDS+=($!)
done
wait "${PIDS[@]}"
for r in "${RUNS[@]}"; do
  IFS='|' read -r tag _ <<< "$r"
  res=$(cat "$O/bytes/$tag/result.txt" 2>/dev/null || echo "RESULT tag=$tag (none)")
  if [ -d "$REF/$tag/out" ] && [ -d "$O/bytes/$tag/out" ]; then
    cmp=$("${C[@]}" compare "$REF/$tag/out" "$O/bytes/$tag/out" 2>&1 | grep -a 'bytes_identical=' | head -1)
    case "$cmp" in *bytes_identical=True*) v=IDENTICAL ;; *) v=DIFFER; FAIL=1 ;; esac
  else
    v=MISSING; cmp="(no output, or no native reference at $REF/$tag/out)"; FAIL=1
  fi
  case "$res" in *" rc=0 "*) ;; *) FAIL=1 ;; esac
  printf '  %-60s %-9s %s\n' "$res" "$v" "$(echo "$cmp" | sed -E 's/^ *//' | cut -c1-150)"
done
fi

# ---------------------------------------------------------------------------------------------------- 3. fetch (collect)
if has fetch; then
note "3. boltz2 fetch-weights (Hugging Face, sha256 against the kit's pins), then against the native copies"
wait "$FETCH_PID"
grep -a -E '^\[fetch\]|^rc=' "$O/fetch.log" | sed 's/^/  /'
grep -q '^rc=0$' "$O/fetch.log" || FAIL=1
for f in boltz2_conf.ckpt boltz2_aff.ckpt ccd.pkl mols.tar; do
  if cmp -s "$L/weights/$f" "$WEIGHTS/$f"; then echo "  $f = native"; else echo "  $f DIFFERS from the native copy (or is missing)"; FAIL=1; fi
done
fi

note "VALIDATE $([ "$FAIL" -eq 0 ] && echo PASS || echo FAILED)  (job ${SLURM_JOB_ID:-local}, $(date +%T))"
echo "outputs in $O"
exit "$FAIL"
