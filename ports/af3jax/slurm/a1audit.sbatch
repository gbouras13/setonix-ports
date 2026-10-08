#!/bin/bash
#SBATCH --job-name=af3jax-A1
#SBATCH --partition=gpu-dev
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:30:00
#SBATCH --output=af3jax-A1.%j.out

# A1 (AF3JAX_PERF_PLAN.md, Status): THE AUDIT OF THE DEFAULT STACK, the reference to rerun after the maintenance. Every piece was
# validated alone (V1: setups A/B/C on the official parameters; G2: rocblas; AC1: one-call triangle attention); here they run together,
# through scripts/af3jax_predict.sh with --regime auto and no other speed flags, against the plain network on the same inputs, all on
# the official AlphaFold 3 parameters. The auto runs share ONE fresh compile cache (as eight lanes sharing $AF3JAX_W/cache would); the
# plain runs compile privately. One GCD per lane, then the whole node:
#   GCD0 Mauto -> Mauto2 -> Mof3   monomer (10 recycles, 2 seeds) auto; again (warm cache: must be bit-identical); auto on the DEFAULT
#                                  weights (OpenFold3-preview2, no --model_dir): the other weights path still works
#   GCD1 Mplain -> B11             monomer, stock1; then the 11-mer (bucket 4,352, 1 recycle, 1 seed) auto (-> B, library GEMMs with
#                                  PERF_PAIRLOGITS_ROWS=1024 since v13: G3's failure, fixed by F1); must fold every chain, not warn
#   GCD2 Qauto / GCD3 Qplain       4-mer (bucket 1,664, 10 recycles, 2 seeds), auto / stock1
#   GCD4 Sauto -> Oauto -> Bgt     6-mer (bucket 2,432, 1 recycle, 2 seeds) auto; 8-mer (3,200, 1 seed) auto; 10-mer setup B with
#                                  --gemm triton (private cache): the control for B's rocblas default (first A1, 50350388: 1.20x)
#   GCD5 Splain / GCD6 Oplain      6-mer / 8-mer, stock1
#   GCD7 Bauto                     10-mer (bucket 3,968, 1 seed) auto (-> B, rocblas)
#   then 8 GCDs: Cplain (stock8) and Cauto (auto -> C), the 12-mer (bucket 4,736, 1 seed)
# Then: the wrapper's automatic choices against the expected ones (regime, GEMM library, attention rows per call, weights); verdicts,
# crashes by name, peaks; the cache check; perf_cmp of each auto run against the plain run of the same input; one speed table.
#
# Run (environment, then sbatch --account=<project>-gpu a1audit.sbatch):
#   A1_DATA     the test data: inputs_ceiling/hk97_{6,8,10,12}mer, inputs_perf/hk97_{mono,4mer}_2s, ref/1OHG.cif (not in this repository)
#   AF3_PARAMS  the official AlphaFold 3 parameters (af3.bin.zst, sha256 checked; never redistributed: bring your own)
#   AF3JAX_SIF  the image (pull_image.sbatch): every run goes through `af3jax predict` in it. Unset: the native tree AF3JAX_W instead.
#   OF3_WEIGHTS the directory with of3_ported_weights.bin.zst, for Mof3 (default $A1_DATA/weights/p2; bound into the image)
#   A1_REF      optional: an earlier A1's output directory (e.g. the native run's): every run is then also compared, sample by sample,
#               with the same run there (same inputs, same seeds) -- the container-against-native check
#   A1_OUT      where the results go (default $A1_DATA)
set -uo pipefail
: "${A1_DATA:?set A1_DATA to the test data directory}" "${AF3_PARAMS:?set AF3_PARAMS to the official parameters directory}"
IC=$A1_DATA/inputs_ceiling; MN=$A1_DATA/inputs_perf/hk97_mono_2s; M4=$A1_DATA/inputs_perf/hk97_4mer_2s
XTAL=$A1_DATA/ref/1OHG.cif
AF3P=$AF3_PARAMS
SHA=296f87576b3aa3813be469eda71d671c598c7a86e9489f49790f6a8e40650c44
OUT=${A1_OUT:-$A1_DATA}/out_A1.$SLURM_JOB_ID
CACHE=$OUT/cache_auto
mkdir -p "$OUT" "$CACHE"
echo "=== node: $(hostname)  job: $SLURM_JOB_ID  partition: ${SLURM_JOB_PARTITION:-?}  $(date) ==="
if [ -n "${AF3JAX_SIF:-}" ]; then
  MODE=container
  OF3W=${OF3_WEIGHTS:-$A1_DATA/weights/p2}
  module load singularity/4.1.0-nohost > /dev/null 2>&1 || { echo "ABORT: no singularity/4.1.0-nohost module"; exit 3; }
  SX=(singularity exec -B "$OF3W:/opt/af3jax/weights/p2" "$AF3JAX_SIF")   # never --rocm (the host's ROCm 6.3.0 libraries)
  PRED=("${SX[@]}" af3jax predict); PY=("${SX[@]}" af3jax python); SCRIPTS=/opt/af3jax/scripts
  NEED=("$AF3JAX_SIF" "$OF3W/of3_ported_weights.bin.zst")
else
  MODE=native
  W=${AF3JAX_W:?set AF3JAX_SIF (container) or AF3JAX_W (native tree)}
  PRED=(bash "$W/scripts/af3jax_predict.sh"); PY=("$W/stockenv/bin/python"); SCRIPTS=$W/scripts
  NEED=("$W/stockenv/bin/python" "$W/scripts/af3jax_predict.sh" "$W/scripts/perf_launch.py" "$W/scripts/rowpair_launch.py" "$W/scripts/perf_cmp.py"
        "$W/scripts/compare_structs.py" "$W/scripts/patch_rocm_capability.py" "$W/scripts/patch_tokamax_rocm.py" "$W/weights/p2/of3_ported_weights.bin.zst")
fi
for f in "${NEED[@]}" "$XTAL" "$AF3P/af3.bin.zst" \
         "$IC/hk97_6mer/hk97_6mer.json" "$IC/hk97_8mer/hk97_8mer.json" "$IC/hk97_10mer/hk97_10mer.json" "$IC/hk97_12mer/hk97_12mer.json" \
         "$MN/hk97_mono_2s.json" "$M4/hk97_4mer_2s.json"; do
  [ -e "$f" ] || { echo "MISSING: $f"; exit 3; }
done
if [ -n "${A1_REF:-}" ]; then [ -d "$A1_REF" ] || { echo "MISSING: A1_REF $A1_REF"; exit 3; }; fi
got=$(sha256sum "$AF3P/af3.bin.zst" | cut -d' ' -f1); [ "$got" = "$SHA" ] || { echo "ABORT: af3.bin.zst sha256 mismatch"; exit 3; }
unset PERF_ATTN_CHUNK PERF_PAIRLOGITS_ROWS AF3JAX_GEMM AF3JAX_CACHE AF3JAX_MODEL_DIR AF3JAX_A_MAX AF3JAX_B_MAX AF3JAX_ROCBLAS_A_MAX \
      AF3JAX_ROCBLAS_B_MAX AF3JAX_ATTN_CHUNK_MAX AF3JAX_PAIRLOGITS_MIN
export TMPDIR=/tmp/af3jax.$SLURM_JOB_ID; mkdir -p "$TMPDIR"
MON_PID=""
trap '[ -n "$MON_PID" ] && kill $MON_PID 2>/dev/null; rm -rf "$TMPDIR"' EXIT
t0=$SECONDS
if [ "$MODE" = container ]; then
  cat "$AF3JAX_SIF" > /dev/null 2>&1
else
  find "$W/stockenv/lib/python3.12/site-packages" -name '*.so' -exec cat {} + > /dev/null 2>&1
fi
cat "$AF3P/af3.bin.zst" > /dev/null 2>&1
echo "pre-warm $((SECONDS-t0)) s"
if [ "$MODE" = container ]; then
  echo "mode: container $AF3JAX_SIF"; "${SX[@]}" af3jax version | sed 's/^/  /'
  "${SX[@]}" af3jax check > "$OUT/check.log" 2>&1 || { echo "ABORT: af3jax check failed in the image:"; tail -20 "$OUT/check.log"; exit 3; }
  echo "af3jax check: PASS ($OUT/check.log)"
else
  echo "mode: native $W"
  echo "md5: wrapper $(md5sum < "$W/scripts/af3jax_predict.sh" | cut -c1-8)  perf_launch $(md5sum < "$W/scripts/perf_launch.py" | cut -c1-8)  rowpair_launch $(md5sum < "$W/scripts/rowpair_launch.py" | cut -c1-8)"
fi
IN=$OUT/inputs
"${PY[@]}" - "$IC/hk97_6mer/hk97_6mer.json" "$IN" "$IC/hk97_10mer/hk97_10mer.json" <<'PYEOF' || { echo "ABORT: inputs"; exit 3; }
import json, os, string, sys
d = dict(json.load(open(sys.argv[1])), name="hk97_6mer_2s", modelSeeds=[1, 2])
os.makedirs(os.path.join(sys.argv[2], "hk97_6mer_2s"), exist_ok=True)
json.dump(d, open(os.path.join(sys.argv[2], "hk97_6mer_2s", "hk97_6mer_2s.json"), "w"))
print("input hk97_6mer_2s: seeds [1, 2]")
d = json.load(open(sys.argv[3]))   # the 11-mer as N1 and N1b built it: the 10-mer with one more copy of its chain
assert len(d["sequences"]) == 1 and len(d["sequences"][0]["protein"]["id"]) == 10, "unexpected 10-mer input"
d = dict(d, name="hk97_11mer")
d["sequences"][0]["protein"]["id"] = list(string.ascii_uppercase[:11])
os.makedirs(os.path.join(sys.argv[2], "hk97_11mer"), exist_ok=True)
json.dump(d, open(os.path.join(sys.argv[2], "hk97_11mer", "hk97_11mer.json"), "w"))
print(f"input hk97_11mer: chains {d['sequences'][0]['protein']['id']}, seeds {d.get('modelSeeds')}")
PYEOF
( while :; do echo "$(date +%T) $(free -g | awk '/^Mem:/ {print "used=" $3 "G avail=" $7 "G total=" $2 "G"}')"; sleep 60; done ) > "$OUT/hostmem.log" 2>&1 &
MON_PID=$!; disown "$MON_PID" 2>/dev/null

pred() {   # pred NAME GCD REGIME INPUT_DIR RECYCLES TIMEOUT WEIGHTS(af3|of3) CACHE(shared|private) [extra wrapper flags...]
  local name=$1 gcd=$2 regime=$3 in=$4 rec=$5 to=$6 wts=$7 cache=$8; shift 8
  local wflag=(); [ "$wts" = af3 ] && wflag=(--model_dir "$AF3P")
  local cflag=(--cache_dir private); [ "$cache" = shared ] && cflag=(--cache_dir "$CACHE")
  echo "launch $name: $(basename "$in") regime=$regime gcd=$gcd r$rec weights=$wts cache=$cache${*:+ $*} ($(date +%T))"
  timeout -k 60 "$to" "${PRED[@]}" --input_dir "$in" --output_dir "$OUT/$name" --regime "$regime" --gcd "$gcd" --num_recycles "$rec" \
    "${wflag[@]}" "${cflag[@]}" "$@" > "$OUT/$name.log" 2>&1
  echo $? > "$OUT/$name.rc"
  echo "done   $name rc=$(cat "$OUT/$name.rc") ($(date +%T))"
}
passed() { [ "$(cat "$OUT/$1.rc" 2>/dev/null)" = 0 ] && grep -q "Done running" "$OUT/$1.log" 2>/dev/null; }
LANES=()
( pred Mauto 0 auto "$MN" 10 1500 af3 shared; pred Mauto2 0 auto "$MN" 10 1500 af3 shared; pred Mof3 0 auto "$MN" 10 1500 of3 shared ) & LANES+=($!)
( pred Mplain 1 stock1 "$MN" 10 1800 af3 private; pred B11 1 auto "$IN/hk97_11mer" 1 4500 af3 shared ) & LANES+=($!)
pred Qauto  2 auto   "$M4" 10 3000 af3 shared & LANES+=($!)
pred Qplain 3 stock1 "$M4" 10 4500 af3 private & LANES+=($!)
( pred Sauto 4 auto "$IN/hk97_6mer_2s" 1 2400 af3 shared; pred Oauto 4 auto "$IC/hk97_8mer" 1 2400 af3 shared
  pred Bgt 4 B "$IC/hk97_10mer" 1 4200 af3 private --gemm triton ) & LANES+=($!)
pred Splain 5 stock1 "$IN/hk97_6mer_2s" 1 4200 af3 private & LANES+=($!)
pred Oplain 6 stock1 "$IC/hk97_8mer" 1 4200 af3 private & LANES+=($!)
pred Bauto  7 auto   "$IC/hk97_10mer" 1 4500 af3 shared & LANES+=($!)
wait "${LANES[@]}"
sleep 15
pred Cplain all stock8 "$IC/hk97_12mer" 1 2700 af3 private
sleep 15
pred Cauto  all auto   "$IC/hk97_12mer" 1 2400 af3 shared

RUNS=(Mauto Mauto2 Mof3 Mplain Qauto Qplain Sauto Splain Oauto Oplain Bauto Bgt B11 Cauto Cplain)
# expected: regime, GEMM library, attention rows per call, pair-logits lever rows, weights
declare -A EXPECT=( [Mauto]="A rocblas none off af3" [Mauto2]="A rocblas none off af3" [Mof3]="A rocblas none off of3"
                    [Qauto]="A rocblas none off af3" [Sauto]="A rocblas none off af3" [Oauto]="A rocblas fork off af3"
                    [Bauto]="B rocblas fork off af3" [Cauto]="C library fork off af3" [Mplain]="stock1 triton fork off af3"
                    [Qplain]="stock1 triton fork off af3" [Splain]="stock1 triton fork off af3" [Oplain]="stock1 triton fork off af3"
                    [Bgt]="B triton fork off af3" [B11]="B rocblas fork 1024 af3" [Cplain]="stock8 library fork off af3" )
echo; echo "=== RESULTS: verdict, then the wrapper's choice (regime gemm attn_chunk weights) against the expected one ==="
SUMMARY=()
for n in "${RUNS[@]}"; do
  log=$OUT/$n.log; rc=$(cat "$OUT/$n.rc" 2>/dev/null || echo "?")
  res=FAIL; passed "$n" && res=PASS
  tag=""
  grep -qE "RESOURCE_EXHAUSTED.*(Out of memory|out of memory)|Out of memory while trying" "$log" 2>/dev/null \
    && tag="$tag DEVICE_OOM($(grep -oE "allocate [0-9.]+ ?[GMK]i?B|allocate [0-9]+ bytes" "$log" | head -1))"
  grep -qE "Segmentation fault|Memory access fault|Fatal Python error|core dumped" "$log" 2>/dev/null && tag="$tag CRASH"
  grep -qE "Cannot find the function" "$log" 2>/dev/null && tag="$tag CRASH_A_SIGNATURE"
  grep -qE "not supported by cublasLt|legacy cublas fallback is removed" "$log" 2>/dev/null && tag="$tag LIBRARY_REFUSED"
  grep -q "WARNING: gemm=" "$log" 2>/dev/null && tag="$tag GEMM_WARNING(library GEMMs above 3,968 on one GCD)"
  case "$rc" in 124|137) tag="$tag TIMEOUT";; esac
  dl=$(grep -m1 "\[af3jax-predict\] tokens" "$log" 2>/dev/null)
  got="$(echo "$dl" | grep -oE "regime=[A-Za-z0-9]+" | cut -d= -f2) $(echo "$dl" | grep -oE "gemm=[a-z]+" | cut -d= -f2) $(echo "$dl" | grep -oE "attn_chunk=[a-z0-9]+" | cut -d= -f2) $(echo "$dl" | grep -oE "pairlogits_rows=[a-z0-9]+" | cut -d= -f2) $(echo "$dl" | grep -oE "weights=(af3|of3)" | cut -d= -f2)"
  choice=OK; [ "$got" = "${EXPECT[$n]}" ] || choice="MISMATCH(expected ${EXPECT[$n]})"
  secs=$(grep -oE "Running model inference with seed [0-9]+ took [0-9.]+" "$log" 2>/dev/null | awk '{printf " s%s=%.1f", $6, $8}')
  pk=$(grep -oE "\[peak\] device=[0-9]+ .*peak_GiB=[0-9.]+" "$log" 2>/dev/null | grep -oE "peak_GiB=[0-9.]+" | cut -d= -f2 | sort -n | tail -1)
  line=$(printf '%-6s %-5s rc=%-4s choice:%-22s %s seeds:%-24s peak=%s GiB%s' "$n" "$res" "$rc" "$got" "$choice" "${secs:- none}" "${pk:-na}" "$tag")
  echo "RESULT $line"; SUMMARY+=("$line")
  echo "$dl" | sed 's/\[af3jax-predict\] /  /' | cut -c1-330
  [ "$res" = PASS ] || grep -nE "Error|error:|Traceback|RESOURCE_EXHAUSTED|Fatal|Segmentation|Memory access|Refus|refus" "$log" 2>/dev/null \
    | grep -v "No AMD" | tail -6 | cut -c1-240 | sed 's/^/  /'
done
echo; echo "=== host memory: $(awk -F'used=' '{split($2, a, "G"); if (a[1] + 0 > m) m = a[1] + 0} END {print "max used " m "G"}' "$OUT/hostmem.log") ==="
echo "=== the shared compile cache: $(du -sh "$CACHE" 2>/dev/null | cut -f1), $(find "$CACHE" -type f 2>/dev/null | wc -l) files; tokamax_autotune.json $([ -f "$CACHE/tokamax_autotune.json" ] && echo present || echo absent)"
echo "    Mauto seed 1 (cold) $(grep -oE "inference with seed 1 took [0-9.]+" "$OUT/Mauto.log" | awk '{print $6}') s; Mauto2 seed 1 (warm cache) $(grep -oE "inference with seed 1 took [0-9.]+" "$OUT/Mauto2.log" | awk '{print $6}') s"
echo; echo "=== PER-CHAIN: core (residues 130-384) CA RMSD of every chain to 1OHG chain A, mean over samples (G3's silent misfold sat near 20 A) ==="
"${PY[@]}" - "$OUT" "$XTAL" "$SCRIPTS" "${RUNS[@]}" <<'PYEOF'
import glob, os, sys
import numpy as np
out, xtal, scripts, runs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
sys.path.insert(0, scripts)
from compare_structs import ca_table, kabsch_rmsd
ref = {n: (v[0], v[1]) for (c, n), v in ca_table(xtal, None, 130, 384, keep={"A"}).items()}
bad = []
for r in runs:
    cifs = sorted(glob.glob(os.path.join(out, r, "*", "seed-*_sample-*", "*_model.cif")))
    if not cifs:
        print(f"  {r:<7} no structures"); bad.append(r); continue
    per = {}
    for cif in cifs:
        t = ca_table(cif, None, 130, 384)
        for c in sorted({c for c, _ in t}):
            ns = [n for (cc, n) in t if cc == c and n in ref and t[(c, n)][0] == ref[n][0]]
            P = np.array([t[(c, n)][1] for n in ns]); Q = np.array([ref[n][1] for n in ns])
            per.setdefault(c, []).append(kabsch_rmsd(P, Q)[0])
    means = {c: float(np.mean(v)) for c, v in per.items()}
    worst = max(means.values())
    ok = worst < 4.0
    bad += [] if ok else [r]
    print(f"  {r:<7} {'FOLDED' if ok else 'MISFOLDED':<9} {len(means)} chains, worst {worst:5.2f} A, mean {np.mean(list(means.values())):5.2f} A")
print(f"  per-chain verdict: {'all folded' if not bad else 'MISFOLDED: ' + ', '.join(bad)}")
PYEOF

CMP=("${PY[@]}" "$SCRIPTS/perf_cmp.py" "$OUT")
X=(--core 130-384 --xtal "$XTAL:A" --conf)
cmp_if() {
  local title=$1 ref=$2; shift 2
  echo; echo "=== $title ==="
  if passed "$ref"; then
    local oth=(); for n in "$@"; do passed "$n" && oth+=("$n"); done
    if [ ${#oth[@]} -gt 0 ]; then "${CMP[@]}" "$ref" "${oth[@]}" "${X[@]}" 2>&1; else echo "(nothing else passed)"; fi
  else
    echo "($ref did not pass: no comparison)"
  fi
}
cmp_if "monomer, 10 recycles, 2 seeds: Mplain = plain (reference); Mauto = defaults; Mauto2 = defaults again from the warm cache" Mplain Mauto Mauto2
cmp_if "monomer: Mauto against Mauto2 (the cache must reproduce it exactly: core 0.00)" Mauto Mauto2
cmp_if "monomer on the default OpenFold3 weights: Mof3 against the official parameters' Mauto (different weights: information only)" Mauto Mof3
cmp_if "4-mer, 10 recycles, 2 seeds: Qplain = plain (reference); Qauto = defaults" Qplain Qauto
cmp_if "6-mer, 1 recycle, 2 seeds: Splain = plain (reference); Sauto = defaults" Splain Sauto
cmp_if "8-mer, 1 recycle, 1 seed: Oplain = plain (reference); Oauto = defaults" Oplain Oauto
cmp_if "10-mer, setup B, 1 recycle, 1 seed: Bgt = setup B with --gemm triton (reference, the default before 2026-10-04); Bauto = defaults (rocblas)" Bgt Bauto
echo; echo "=== 11-mer (bucket 4,352), B with library GEMMs and the pair-logits lever: no plain run fits here; its RESULT line and its per-chain verdict are the check (F1: 2.3-2.5 A per chain, 53.13 GiB) ==="
cmp_if "12-mer on 8 GCDs, 1 recycle, 1 seed: Cplain = plain row-sharded (reference); Cauto = defaults (setup C)" Cplain Cauto

if [ -n "${A1_REF:-}" ]; then
  echo; echo "=== AGAINST A1_REF ($A1_REF): every run against the same run there (same input, same seeds; REF = that run) ==="
  for n in "${RUNS[@]}"; do
    if passed "$n" && [ -d "$A1_REF/$n" ] && [ -f "$A1_REF/$n.log" ]; then
      ln -sfn "$A1_REF/$n" "$OUT/ref_$n"; ln -sf "$A1_REF/$n.log" "$OUT/ref_$n.log"
      echo "--- $n"; "${CMP[@]}" "ref_$n" "$n" "${X[@]}" 2>&1
    else
      echo "--- $n: not compared (it did not pass here, or A1_REF has no $n)"
    fi
  done
fi

echo; echo "=== SPEED: defaults against the plain network (warm seed where there are two, otherwise the cold seed) ==="
"${PY[@]}" - "$OUT" <<'PYEOF'
import os, re, sys
out = sys.argv[1]
def times(n):
    try:
        t = dict(re.findall(r"Running model inference with seed (\d+) took ([0-9.]+)", open(os.path.join(out, n + ".log"), errors="replace").read()))
    except OSError:
        return None
    return float(t["2"]) if "2" in t else (float(t["1"]) if "1" in t else None)
for label, plain, auto in (("monomer, 10 recycles", "Mplain", "Mauto"), ("4-mer, 10 recycles", "Qplain", "Qauto"), ("6-mer, 1 recycle", "Splain", "Sauto"),
                           ("8-mer, 1 recycle", "Oplain", "Oauto"), ("12-mer, 8 GCDs", "Cplain", "Cauto"), ("10-mer B: triton -> rocblas", "Bgt", "Bauto")):
    p, a = times(plain), times(auto)
    print(f"  {label:<28} {plain:>7} {p if p else '-':>9}  {auto:>7} {a if a else '-':>9}  " + (f"{p / a:5.2f}x" if p and a else "  -"))
PYEOF
echo; echo "=== SUMMARY ==="
printf '  %s\n' "${SUMMARY[@]}"
echo "=== END $(date) ==="
