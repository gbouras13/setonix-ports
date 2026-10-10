#!/bin/bash
#SBATCH --job-name=colabfold-validate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8,tmp:1000G
#SBATCH --exclusive
#SBATCH --time=01:15:00
#SBATCH --output=colabfold-validate.%j.out

# The colabfold image against the native port, on one Setonix GPU node (8 MI250X GCDs, the node's NVMe booked for the databases).
#   sbatch --account=<project>-gpu validate.sbatch SIF=<the .sif> OUT=<dir> NATIVE_W=<the native colabfold-setonix tree> \
#          REF_PRED=<the native bitwise check's output dir> NATIVE_MMSEQS=<dir with the native bin/mmseqs, mmseqs-dbload2, msa_gpu.sbatch> \
#          [DB=<ColabFold database root>]                         (arguments, not variables: Setonix sets SBATCH_EXPORT=NONE)
# 1. `colabfold check` on all 8 GCDs: devices, a matmul each, every GPU library from the image, the nvidia-smi shim, MMseqs2 on the GPU.
# 2. Predictions, byte for byte against REF_PRED (the native bitwise check, job 50480325): 1BRS off and exact (2 models), the HK97 trimer
#    off and exact (1 model); XLA's deterministic flags, copy-only serialisation, seed 0, 3 recycles. One GCD each, all at once; then the
#    port's cf_outputs.py compare (structures and scores, also when not byte-identical).
# 3. MSAs for the 246 SAOMS1 proteins, with templates: natively (msa_gpu.sbatch: the native binary and colabfold_search, NVMe staging,
#    gpuservers), then through the image (colabfold search, reusing the staged databases); every a3m and m8 compared byte for byte.
set -uo pipefail
for a in "$@"; do
  case "$a" in
    SIF=*|OUT=*|NATIVE_W=*|REF_PRED=*|NATIVE_MMSEQS=*|DB=*) export "${a?}" ;;
    *) echo "unknown argument '$a': SIF= OUT= NATIVE_W= REF_PRED= NATIVE_MMSEQS= DB="; exit 2 ;;
  esac
done
: "${SIF:?SIF=}" "${OUT:?OUT=}" "${NATIVE_W:?NATIVE_W=}" "${REF_PRED:?REF_PRED=}" "${NATIVE_MMSEQS:?NATIVE_MMSEQS=}"
DB=${DB:-/scratch/references/colabfold_jun2026/database}
O=$OUT/validate.${SLURM_JOB_ID:-local}
mkdir -p "$O" && cd "$O" || exit 3
for f in "$SIF" "$NATIVE_W/venv/bin/colabfold_search" "$NATIVE_W/inputs/hk97/hk97_k3.a3m" "$NATIVE_MMSEQS/bin/mmseqs" "$NATIVE_MMSEQS/mmseqs-dbload2" \
         "$NATIVE_MMSEQS/msa_gpu.sbatch" "$REF_PRED/D1_off_det_A/out" "$REF_PRED/D2_exact_det/out" "$REF_PRED/D3_off_det_hk97_A/out" \
         "$REF_PRED/D4_exact_det_hk97/out"; do
  [ -e "$f" ] || { echo "MISSING: $f"; exit 3; }
done
module load singularity/4.1.0-nohost > /dev/null 2>&1 || { echo "ABORT: no singularity/4.1.0-nohost module"; exit 3; }
export SINGULARITY_CACHEDIR=/tmp/sing-cache.${SLURM_JOB_ID:-$$} APPTAINER_CACHEDIR=/tmp/sing-cache.${SLURM_JOB_ID:-$$}
C=(singularity exec "$SIF" colabfold)    # never --rocm
echo "=== node $(hostname)  job ${SLURM_JOB_ID:-?}  $(date)  image $SIF ($(du -h "$SIF" | cut -f1)) ==="
"${C[@]}" version | sed 's/^/  /'
t0=$SECONDS; cat "$SIF" > /dev/null; echo "pre-read the image: $((SECONDS - t0)) s"

echo; echo "=== 1. colabfold check (all GCDs) ==="
"${C[@]}" check > check.log 2>&1; rc=$?
grep -E "^\[check\] (versions|pins|devices|matmul|shim|mmseqs|libs|driver|FAIL|PASS)" check.log | cut -c1-200
[ $rc -eq 0 ] || { echo "CHECK FAILED (rc $rc): see $O/check.log"; tail -20 check.log; }

echo; echo "=== 2. predictions against the native bitwise check ==="
IN1=/opt/colabfold/kit/colabfold/tests/inputs/1BRS_AD.a3m      # inside the image (the kit's own test input)
IN3=$NATIVE_W/inputs/hk97/hk97_k3.a3m
SET2=(--model-type alphafold2_multimer_v3 --num-models 2 --num-recycle 3 --recycle-early-stop-tolerance 0 --random-seed 0)
SET1=(--model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 3 --recycle-early-stop-tolerance 0 --random-seed 0)
export CFS_XLA_EXTRA="--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true"
pred() {   # pred NAME GCD MODE INPUT colabfold_batch options...
  local name=$1 gcd=$2 mode=$3 in=$4; shift 4
  mkdir -p "$O/$name"
  ( cd "$O/$name" && ROCR_VISIBLE_DEVICES=$gcd timeout -k 60 1500 "${C[@]}" predict --mode "$mode" "$in" out "$@" > run.log 2>&1
    echo $? > rc )
}
t0=$SECONDS
pred D1_off_det 0 off "$IN1" "${SET2[@]}" &
pred D2_exact_det 1 exact "$IN1" "${SET2[@]}" &
pred D3_off_det_hk97 2 off "$IN3" "${SET1[@]}" &
pred D4_exact_det_hk97 3 exact "$IN3" "${SET1[@]}" &
wait
echo "predictions done in $((SECONDS - t0)) s"
same=0; diff=0
for pair in D1_off_det:D1_off_det_A D2_exact_det:D2_exact_det D3_off_det_hk97:D3_off_det_hk97_A D4_exact_det_hk97:D4_exact_det_hk97; do
  ours=${pair%%:*}; ref=${pair#*:}
  echo "-- $ours: rc=$(cat "$O/$ours/rc" 2>/dev/null)  $(grep -a -o -E '\[cfs\] input=[^ ]+ tokens=[0-9]+ -> serialise=[a-z]+' "$O/$ours/run.log" | head -1)"
  for f in "$REF_PRED/$ref"/out/*_rank_*.pdb "$REF_PRED/$ref"/out/*_scores_rank_*.json; do
    g=$O/$ours/out/$(basename "$f")
    if [ ! -f "$g" ]; then r=MISSING; diff=$((diff + 1)); elif cmp -s "$f" "$g"; then r=identical; same=$((same + 1)); else r=DIFFERENT; diff=$((diff + 1)); fi
    echo "   $(basename "$f"): $r"
  done
  "${C[@]}" python /opt/colabfold/tools/cf_outputs.py compare "$REF_PRED/$ref/out" "$O/$ours/out" 2>&1 | sed 's/^/   /' | head -8
done
echo "PREDICTIONS: $same files byte-identical to the native run, $diff different or missing"

echo; echo "=== 3. MSAs: native msa_gpu.sbatch, then the image's colabfold search (246 SAOMS1 proteins, templates) ==="
Q=$O/SAOMS1_phanotate.faa
singularity exec "$SIF" cat /opt/colabfold/tests/SAOMS1_phanotate.faa > "$Q"
t0=$SECONDS
( export MMSEQS=$NATIVE_MMSEQS/bin/mmseqs MMSEQS_WRAP=$NATIVE_MMSEQS/mmseqs-dbload2 CFSEARCH=$NATIVE_W/venv/bin/colabfold_search COLABFOLD_DB=$DB
  bash "$NATIVE_MMSEQS/msa_gpu.sbatch" "$Q" "$O/msa_native" --use-templates 1 ) > msa_native.log 2>&1
echo "native: rc=$?  $((SECONDS - t0)) s  $(grep -E '^staged|colabfold_search:' msa_native.log | tr '\n' ' ')"
touch "${TMPDIR:-/tmp}/cfdb-${SLURM_JOB_ID:-$$}/.staged"      # the native run staged the same files under the same name: the image reuses them
t0=$SECONDS
STAGE=1 THREADS=64 COLABFOLD_DB=$DB "${C[@]}" search "$Q" "$O/msa_image" --use-templates 1 > msa_image.log 2>&1   # msa_gpu.sbatch's 64 threads
echo "image:  rc=$?  $((SECONDS - t0)) s  $(grep -E 'cf-search: (staged|colabfold_search)' msa_image.log | tr '\n' ' ')"
same=0; diff=0; miss=0
for f in "$O"/msa_native/*.a3m "$O"/msa_native/*.m8; do
  [ -e "$f" ] || continue
  g=$O/msa_image/$(basename "$f")
  if [ ! -f "$g" ]; then miss=$((miss + 1)); echo "   missing in the image's: $(basename "$f")"
  elif cmp -s "$f" "$g"; then same=$((same + 1))
  else diff=$((diff + 1)); [ $diff -le 5 ] && echo "   DIFFERENT: $(basename "$f") ($(grep -c '^>' "$f" 2>/dev/null) vs $(grep -c '^>' "$g" 2>/dev/null) records)"; fi
done
extra=$(comm -13 <(cd "$O/msa_native" && ls ./*.a3m ./*.m8 2>/dev/null | sort) <(cd "$O/msa_image" && ls ./*.a3m ./*.m8 2>/dev/null | sort) | wc -l)
echo "MSAS: $same files byte-identical, $diff different, $miss missing from the image's, $extra only in the image's"
echo; echo "=== END $(date) ==="
