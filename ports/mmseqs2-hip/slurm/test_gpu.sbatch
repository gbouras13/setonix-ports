#!/bin/bash
#SBATCH --job-name=mmseqs2-hip-test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=mmseqs2-hip-test.%j.out

# BUILD.md's correctness test (job 50232402) on one Setonix GPU node (8 MI250X GCDs), for the image or for a binary you built:
#   sbatch --account=<project>-gpu test_gpu.sbatch SIF=<the .sif> OUT=<dir> [NATIVE=<a native mmseqs>] [DB=<ColabFold database root>]
#   sbatch --account=<project>-gpu test_gpu.sbatch BIN=<your mmseqs> QUERIES=<tests/SAOMS1_phanotate.faa> OUT=<dir> [DB=...]
#   (arguments, not variables: Setonix sets SBATCH_EXPORT=NONE)
# Queries: the 246 proteins of Staphylococcus phage SAOMS1 (pharokka test data: tests/SAOMS1_phanotate.faa, in the image at
# /opt/mmseqs/tests).
#   1. pdb100 (GPU-padded): the CPU's ungapped prefilter (the algorithm the GPU runs) against the GPU on 1 GCD, on 8 GCDs and through
#      gpuserver; the CPU's default k-mer prefilter for reference. Hits compared pair by pair.
#   2. UniRef30 2302 consensus (GPU-padded, 8.7 GB): GPU on 8 GCDs against the CPU, 10 queries.
#   3. NATIVE given: the native binary's GPU searches of 1 and 2 against the image's. The same hits and scores are expected.
set -uo pipefail
for a in "$@"; do
  case "$a" in
    SIF=*|BIN=*|QUERIES=*|OUT=*|NATIVE=*|DB=*) export "${a?}" ;;
    *) echo "unknown argument '$a': SIF= | BIN= QUERIES=, OUT= NATIVE= DB="; exit 2 ;;
  esac
done
: "${OUT:?OUT=<a directory>}"
[ -n "${SIF:-}" ] || [ -n "${BIN:-}" ] || { echo "give SIF=<the mmseqs2-hip .sif> or BIN=<an mmseqs binary>"; exit 2; }
[ -n "${QUERIES:-}" ] && QUERIES=$(readlink -f "$QUERIES")
DB=${DB:-/scratch/references/colabfold_jun2026/database}
PDB=$DB/pdb100/pdb100_230517
UR=$DB/colabfold_uniref30/uniref30_2302_db
W=$OUT/run-${SLURM_JOB_ID:-local}; mkdir -p "$W"; cd "$W" || exit 3
if [ -n "${SIF:-}" ]; then
  module load singularity/4.1.0-nohost > /dev/null 2>&1 || { echo "ABORT: no singularity/4.1.0-nohost module"; exit 3; }
  export SINGULARITY_CACHEDIR=${TMPDIR:-/tmp}/sing-cache.$$ APPTAINER_CACHEDIR=${TMPDIR:-/tmp}/sing-cache.$$
  M=(singularity exec "$SIF" mmseqs)   # never --rocm
  CAT=(singularity exec "$SIF" cat)
  QUERIES=${QUERIES:-/opt/mmseqs/tests/SAOMS1_phanotate.faa}
  echo "node $(hostname)  $(date -Is)  image $SIF"
  singularity exec "$SIF" cat /opt/mmseqs/build.txt /opt/mmseqs/source.txt
else
  M=("$BIN")
  CAT=(cat)
  : "${QUERIES:?QUERIES=<tests/SAOMS1_phanotate.faa>}"
  echo "node $(hostname)  $(date -Is)  binary $BIN"
fi
export OMP_PROC_BIND=false
echo "mmseqs $("${M[@]}" version)"
[ -n "${NATIVE:-}" ] && echo "native $NATIVE: $("$NATIVE" version)"

run () {   # label, then the command; logs to log.<label>, prints wall time and exit code
  local lab=$1; shift
  local t0 t1 rc
  t0=$(date +%s.%N); "$@" > "log.$lab" 2>&1; rc=$?; t1=$(date +%s.%N)
  printf "[test] %-16s rc=%d  %8.1f s\n" "$lab" "$rc" "$(awk -v a="$t0" -v b="$t1" 'BEGIN{print b-a}')"
  [ $rc -eq 0 ] || tail -5 "log.$lab"
  return 0
}
m8 () {    # binary..., query db, target db, result db, out
  local n=$#; local args=("$@")
  "${args[@]:0:$((n - 4))}" convertalis "${args[$((n - 4))]}" "${args[$((n - 3))]}" "${args[$((n - 2))]}" "${args[$((n - 1))]}" \
    --format-output query,target,evalue,bits,pident,alnlen,qstart,qend,tstart,tend > /dev/null 2>&1
}

"${M[@]}" createdb "$QUERIES" qdb > /dev/null
"${CAT[@]}" "$QUERIES" | awk '/^>/{n++} n<=10' > q10.faa
"${M[@]}" createdb q10.faa q10 > /dev/null

echo; echo "=== 1. pdb100, 246 queries ==="
run cpu_kmer     "${M[@]}" search qdb "$PDB" r_cpu0 t_cpu0 --threads 64
run cpu_ungapped "${M[@]}" search qdb "$PDB" r_cpu1 t_cpu1 --threads 64 --prefilter-mode 1
run gpu_1gcd     env HIP_VISIBLE_DEVICES=0 "${M[@]}" search qdb "$PDB" r_g1 t_g1 --threads 16 --gpu 1 --prefilter-mode 1
run gpu_8gcd     "${M[@]}" search qdb "$PDB" r_g8 t_g8 --threads 64 --gpu 1 --prefilter-mode 1
"${M[@]}" gpuserver "$PDB" --prefilter-mode 1 --db-load-mode 0 > log.gpuserver 2>&1 &
SRV=$!
run gpu_server   "${M[@]}" search qdb "$PDB" r_gs t_gs --threads 64 --gpu 1 --gpu-server 1 --gpu-server-wait-timeout 300 --prefilter-mode 1
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
for r in cpu0 cpu1 g1 g8 gs; do m8 "${M[@]}" qdb "$PDB" "r_$r" "pdb.$r.m8"; done

echo; echo "=== 2. UniRef30 2302 consensus (8.7 GB), 10 queries ==="
run ur_gpu8_10   "${M[@]}" search q10 "$UR" ur10_g8 t_ur10_g8 --threads 64 --gpu 1 --prefilter-mode 1
run ur_cpu_10    "${M[@]}" search q10 "$UR" ur10_cpu t_ur10_cpu --threads 64 --prefilter-mode 1
m8 "${M[@]}" q10 "$UR" ur10_g8 ur10.g8.m8
m8 "${M[@]}" q10 "$UR" ur10_cpu ur10.cpu.m8

if [ -n "${NATIVE:-}" ]; then
  echo; echo "=== 3. the native binary, the same GPU searches ==="
  run nat_pdb_8gcd "$NATIVE" search qdb "$PDB" rn_g8 tn_g8 --threads 64 --gpu 1 --prefilter-mode 1
  run nat_ur_8gcd  "$NATIVE" search q10 "$UR" urn10_g8 tn_ur10_g8 --threads 64 --gpu 1 --prefilter-mode 1
  m8 "$NATIVE" qdb "$PDB" rn_g8 pdb.ng8.m8
  m8 "$NATIVE" q10 "$UR" urn10_g8 ur10.ng8.m8
fi

echo; echo "=== comparison (pairs = query,target; identical = the same E-value and bit score) ==="
python3 - <<'EOF'
import os
def load(p):
    d = {}
    if not os.path.exists(p):
        return None
    for l in open(p):
        f = l.rstrip('\n').split('\t')
        d.setdefault((f[0], f[1]), (f[2], f[3]))   # the first (best) alignment per pair
    return d
bad = 0
def cmp(a, b, la, lb, must_match=True):
    global bad
    A, B = load(a), load(b)
    if A is None or B is None:
        if os.path.exists(a) or os.path.exists(b) or must_match:
            print(f"{la} vs {lb}: missing output"); bad += must_match
        return
    common = A.keys() & B.keys()
    same = sum(A[k] == B[k] for k in common)
    ok = len(A) == len(B) == len(common) == same
    print(f"{la:>14} vs {lb:<14} pairs {len(A):6d} / {len(B):6d}  shared {len(common):6d}  only-{la} {len(A.keys() - B.keys()):5d}  "
          f"only-{lb} {len(B.keys() - A.keys()):5d}  identical {same}/{len(common)}  {'IDENTICAL' if ok else ('DIFFERENT' if must_match else '(reference)')}")
    bad += must_match and not ok
cmp('pdb.cpu1.m8', 'pdb.g1.m8', 'cpu_ungapped', 'gpu_1gcd')
cmp('pdb.cpu1.m8', 'pdb.g8.m8', 'cpu_ungapped', 'gpu_8gcd')
cmp('pdb.g8.m8', 'pdb.gs.m8', 'gpu_8gcd', 'gpu_server')
cmp('pdb.cpu0.m8', 'pdb.g8.m8', 'cpu_kmer', 'gpu_8gcd', must_match=False)
cmp('ur10.cpu.m8', 'ur10.g8.m8', 'ur_cpu', 'ur_gpu8')
if os.path.exists('pdb.ng8.m8') or os.path.exists('ur10.ng8.m8'):
    cmp('pdb.ng8.m8', 'pdb.g8.m8', 'native_pdb', 'tested_pdb')
    cmp('ur10.ng8.m8', 'ur10.g8.m8', 'native_ur', 'tested_ur')
print("VERDICT:", "PASS (every GPU search identical to its reference)" if bad == 0 else f"FAIL ({bad} comparison(s) differ)")
EOF
echo; echo "[test] done $(date -Is)"
