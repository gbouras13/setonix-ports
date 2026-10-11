#!/bin/bash
#SBATCH --job-name=proteinmpnn-validate
#SBATCH --partition=gpu-highmem
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=proteinmpnn-validate.%j.out

# The proteinmpnn image against the native port, on one Setonix GPU node (8 MI250X GCDs). gpu-highmem, exclusive, because the throughput
# section is an A/A comparison and a shared node makes the two halves incomparable.
#   sbatch --account=<project>-gpu validate.sbatch SIF=<the .sif> OUT=<dir> NATIVE_W=<the native proteinmpnn-setonix tree> \
#          [PARTS=check,bytes,tput] [BENCH=<dir of PDBs>] [SEQS=8]
#   (arguments, not variables: Setonix sets SBATCH_EXPORT=NONE). PARTS picks the sections, default all three.
#
# 1. check  `proteinmpnn check --design`: versions, the kit's own pin check for both weight sets, every GPU library loaded from inside
#           the image, a matmul on each of the 8 GCDs, torch's uniform transform against rocRAND's and cuRAND's, and -- the --design
#           part -- upstream's 429-residue 3HTN designed BOTH ways inside the image, stock against the kit's exact worker, bytes
#           compared. 429 residues is past the length at which ROCm 7.2's graph packet capture corrupts the heap, so a pass here is
#           also a pass for DEBUG_CLR_GRAPH_PACKET_CAPTURE=0.
# 2. bytes  the same six upstream examples through four routes -- native stock, native exact, image stock, image exact -- at a fixed
#           seed, and every one of the twelve pairings compared byte for byte. This is the question the image has to answer: an
#           interpreter built elsewhere (python-build-standalone 3.11.14, not cray-python 3.11.7) and ROCm libraries pruned to gfx90a
#           must still produce the bytes in af3-setonix/PROTEINMPNN_PORT.md.
# 3. tput   a benchmark pass over BENCH on all 8 GCDs, two workers each, image and native, exact mode, both warmed first. This is
#           an A/A sanity check -- do the two arms agree, and are the designs identical -- not a reproduction of the port's 402
#           designs/s, which was measured over 1,472 backbones; bench92 is 92, where start-up is most of the wall clock.
#           Skipped quietly if BENCH is absent.
set -uo pipefail
for a in "$@"; do
  case "$a" in
    SIF=*|OUT=*|NATIVE_W=*|PARTS=*|BENCH=*|SEQS=*) export "${a?}" ;;
    *) echo "unknown argument '$a': SIF= OUT= NATIVE_W= [PARTS=] [BENCH=] [SEQS=]"; exit 2 ;;
  esac
done
PARTS=${PARTS:-check,bytes,tput}
SEQS=${SEQS:-8}
has() { case ",$PARTS," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
: "${SIF:?SIF=}" "${OUT:?OUT=}" "${NATIVE_W:?NATIVE_W=}"
BENCH=${BENCH:-$NATIVE_W/inputs/bench92}
O=$OUT/validate.${SLURM_JOB_ID:-local}
mkdir -p "$O" && cd "$O" || exit 3
for f in "$SIF" "$NATIVE_W/venv/bin/python" "$NATIVE_W/scripts/pms-env.sh"; do
  [ -e "$f" ] || { echo "MISSING: $f"; exit 3; }
done
module load singularity/4.1.0-nohost > /dev/null 2>&1 || { echo "ABORT: no singularity/4.1.0-nohost module"; exit 3; }
export SINGULARITY_CACHEDIR=/tmp/sing-cache.${SLURM_JOB_ID:-$$} APPTAINER_CACHEDIR=$SINGULARITY_CACHEDIR
export PMS_LOCAL=/tmp/pms-$USER-${SLURM_JOB_ID:-$$}
mkdir -p "$PMS_LOCAL"
C=(singularity exec --bind "$PMS_LOCAL:$PMS_LOCAL" --bind "$O:$O" "$SIF" proteinmpnn)    # never --rocm
FAIL=0
note() { printf '\n######## %s\n' "$*"; }

echo "=== node $(hostname)  job ${SLURM_JOB_ID:-?}  $(date)  image $SIF ($(du -h "$SIF" | cut -f1)) ==="
"${C[@]}" version | sed 's/^/  /'
t0=$SECONDS; cat "$SIF" > /dev/null; echo "pre-read the image: $((SECONDS - t0)) s"

# the native tree's own runner, for the arms that are not in the image
native() {
  ( set -e
    cd "$NATIVE_W"
    . "$NATIVE_W/scripts/pms-env.sh" > /dev/null || exit 97
    exec bash "$NATIVE_W/scripts/pms-run.sh" "$@" ) ;
}

# ---------------------------------------------------------------------------------------------------- 1. check
if has check; then
note "1. proteinmpnn check --design (all 8 GCDs; 3HTN designed stock vs exact inside the image)"
"${C[@]}" check --design --keep "$O/check_design" > check.log 2>&1; rc=$?
sed -n '1,200p' check.log | sed 's/^/  /'
echo "check rc=$rc"; [ "$rc" -eq 0 ] || FAIL=1
fi

# ---------------------------------------------------------------------------------------------------- 2. bytes
if has bytes; then
note "2. six upstream examples, four routes, byte for byte"
EX=$NATIVE_W/ProteinMPNN/inputs
IN=$O/examples; mkdir -p "$IN"
for p in "$EX"/PDB_monomers/pdbs/5L33.pdb "$EX"/PDB_monomers/pdbs/6MRR.pdb "$EX"/PDB_complexes/pdbs/3HTN.pdb \
         "$EX"/PDB_complexes/pdbs/4YOW.pdb "$EX"/PDB_homooligomers/pdbs/4GYT.pdb "$EX"/PDB_homooligomers/pdbs/6EHB.pdb; do
  [ -f "$p" ] && cp -n "$p" "$IN/" || echo "  (absent: $p)"
done
echo "  $(ls "$IN" | wc -l) inputs, seed 37, $SEQS designs each, sampling_temp 0.1"
OPTS=(--seed 37 --num_seq_per_target "$SEQS" --sampling_temp 0.1 --batch_size 1)
# `exact` needs --hybrid_gemm 0, and it has to come before the input: run.sh stops reading its own options at the first it does not
# know. Without it the kit refuses the mode by name (rc 3, nothing designed), because its probe finds that lever is not bit-identical
# on gfx90a. The probe passes on ONE backbone and fails from two, so a single-input arm would not reveal the omission -- 50645953 ran
# six and both exact arms, native and image alike, exited 3.
for arm in native_off native_exact image_off image_exact; do
  mode=${arm#*_}; d=$O/$arm; rm -rf "$d"
  KIT=(); [ "$mode" = exact ] && KIT=(--hybrid_gemm 0)
  t0=$SECONDS
  case "$arm" in
    native_*) native design --mode "$mode" ${KIT[@]+"${KIT[@]}"} --input "$IN" --out "$d" "${OPTS[@]}" > "$arm.log" 2>&1; rc=$? ;;
    image_*)  "${C[@]}" design --mode "$mode" ${KIT[@]+"${KIT[@]}"} --input "$IN" --out "$d" "${OPTS[@]}" > "$arm.log" 2>&1; rc=$? ;;
  esac
  echo "  $(printf '%-13s' "$arm") rc=$rc  $((SECONDS - t0)) s  $(ls "$d/seqs" 2>/dev/null | wc -l) fa"
  [ "$rc" -eq 0 ] || FAIL=1
done
echo
echo "  pairwise (every pair must be identical: the kit's exact line is byte-identical to stock, and the image to the native tree)"
for pair in native_off:native_exact native_off:image_off native_off:image_exact native_exact:image_exact image_off:image_exact; do
  a=${pair%%:*}; b=${pair##*:}
  if [ -d "$O/$a/seqs" ] && [ -d "$O/$b/seqs" ]; then
    out=$("${C[@]}" compare "$O/$a" "$O/$b" --quiet 2>&1); rc=$?
    printf '  %-28s %s  %s\n' "$a vs $b" "$([ $rc -eq 0 ] && echo IDENTICAL || echo DIFFER)" "$(echo "$out" | tail -1)"
    [ "$rc" -eq 0 ] || FAIL=1
  else
    printf '  %-28s SKIPPED (an arm produced nothing)\n' "$a vs $b"; FAIL=1
  fi
done
fi

# ---------------------------------------------------------------------------------------------------- 3. throughput
if has tput; then
note "3. throughput, exact, 8 GCDs x 2 workers (the native port measured 402 designs/s on a node)"
if [ ! -d "$BENCH" ]; then
  echo "  BENCH=$BENCH is not a directory -- skipped"
else
  echo "  $(ls "$BENCH"/*.pdb 2>/dev/null | wc -l) backbones x $SEQS designs"
  # Warm both arms first. The exact worker compiles one Triton kernel (the fused draw); the native tree keeps it in a persistent JIT
  # root on /scratch, while the image defaults to a per-job directory, so a cold image pays a compile per worker that the native tree
  # does not. At this size that is most of the wall clock: in 50645953 the image read 18.1 designs/s against native's 31.2 purely
  # from this. Measure both warm, or measure nothing.
  WARM=$O/warm; rm -rf "$WARM"; mkdir -p "$WARM"
  cp "$(ls "$BENCH"/*.pdb 2>/dev/null | head -1)" "$WARM/" 2>/dev/null
  for arm in image native; do
    case $arm in
      image)  "${C[@]}" design --mode exact --input "$WARM" --out "$O/warm_$arm" --seed 37 --num_seq_per_target 1 ;;
      native) native design --mode exact --hybrid_gemm 0 --input "$WARM" --out "$O/warm_$arm" --seed 37 --num_seq_per_target 1 ;;
    esac > "warm_$arm.log" 2>&1
    echo "  warm-up $arm: rc=$? (compiles the draw kernel; not timed)"
  done
  for arm in image native; do
    d=$O/tput_$arm; rm -rf "$d"; t0=$SECONDS
    case $arm in
      image)  "${C[@]}" dp --mode exact --input "$BENCH" --out "$d" --gcds 0,1,2,3,4,5,6,7 --per-gcd 2 \
                 --seed 37 --num_seq_per_target "$SEQS" --sampling_temp 0.1 > "tput_$arm.log" 2>&1; rc=$? ;;
      native) ( set -e; cd "$NATIVE_W"; . "$NATIVE_W/scripts/pms-env.sh" > /dev/null || exit 97
                exec python "$NATIVE_W/scripts/pms_dp.py" --mode exact --input "$BENCH" --out "$d" \
                     --gcds 0,1,2,3,4,5,6,7 --per-gcd 2 --seed 37 --num_seq_per_target "$SEQS" --sampling_temp 0.1 \
              ) > "tput_$arm.log" 2>&1; rc=$? ;;
    esac
    printf '  %-7s rc=%d  %4d s  %s\n' "$arm" "$rc" "$((SECONDS - t0))" "$(grep -a '^DP-SUMMARY' "tput_$arm.log" | tail -1)"
    [ "$rc" -eq 0 ] || FAIL=1
  done
  if [ -d "$O/tput_image/seqs" ] && [ -d "$O/tput_native/seqs" ]; then
    "${C[@]}" compare "$O/tput_image" "$O/tput_native" --quiet > tput_compare.log 2>&1; rc=$?
    echo "  image vs native over the benchmark: $([ $rc -eq 0 ] && echo IDENTICAL || echo DIFFER)  $(tail -1 tput_compare.log)"
    [ "$rc" -eq 0 ] || FAIL=1
  fi
fi
fi

note "VALIDATE $([ "$FAIL" -eq 0 ] && echo PASS || echo FAILED)  (job ${SLURM_JOB_ID:-local}, $(date +%T))"
echo "outputs in $O"
exit "$FAIL"
