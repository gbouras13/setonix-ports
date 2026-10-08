#!/bin/bash
# test_wrapper.sh -- the af3jax wrapper's decision table, checked with --dry_run against a stand-in tree (no GPU, no AlphaFold 3: only the
# wrapper's own logic). Each case states the input (HK97 k-mers of 385 residues, so tokens = 385 k), the options and the decision the
# validated defaults must produce: setup, GEMM library (with its source), and the pair-logits lever (AF3JAX_PORT.md: G3 / G6 / G7 / F1,
# wrapper v13), plus whether the one-GCD library warning prints. Run: bash ports/af3jax/tests/test_wrapper.sh   (exit 1 on any failure)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WRAPPER=$HERE/../scripts/af3jax_predict.sh
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
W=$T/w
mkdir -p "$W/scripts" "$W/stockenv/bin" "$W/stockenv/lib/python3.12/site-packages" "$W/alphafold" "$W/weights/p2" "$T/in"
cp "$WRAPPER" "$W/scripts/af3jax_predict.sh"
ln -s "$(command -v python3)" "$W/stockenv/bin/python"
touch "$W/alphafold/run_alphafold.py" "$W/weights/p2/of3_ported_weights.bin.zst"
for s in perf_launch.py rowpair_launch.py patch_rocm_capability.py patch_tokamax_rocm.py; do touch "$W/scripts/$s"; done
python3 - "$T/in" <<'EOF'
import json, os, string, sys
for k in (1, 8, 10, 11, 12):
    d = os.path.join(sys.argv[1], f"k{k}"); os.makedirs(d)
    json.dump({"name": f"k{k}", "sequences": [{"protein": {"id": list(string.ascii_uppercase[:k]), "sequence": "M" * 385}}],
               "modelSeeds": [1]}, open(os.path.join(d, f"k{k}.json"), "w"))
EOF

fails=0
# check NAME K EXPECTED [env assignments...] -- [wrapper options...]
#   EXPECTED = "bucket regime gemm(source) pairlogits(source) warn|-"
check() {
  local name=$1 k=$2 want=$3; shift 3
  local envs=() opts=()
  while [ $# -gt 0 ] && [ "$1" != -- ]; do envs+=("$1"); shift; done
  [ "${1:-}" = -- ] && shift
  opts=("$@")
  local out got
  out=$(env -u PERF_ATTN_CHUNK -u PERF_PAIRLOGITS_ROWS -u AF3JAX_GEMM -u AF3JAX_ROCBLAS_A_MAX -u AF3JAX_ROCBLAS_B_MAX -u AF3JAX_PAIRLOGITS_MIN \
        AF3JAX_W="$W" ${envs[@]+"${envs[@]}"} bash "$W/scripts/af3jax_predict.sh" --input_dir "$T/in/k$k" --output_dir "$T/out" --dry_run \
        ${opts[@]+"${opts[@]}"} 2>&1)
  local line; line=$(printf '%s\n' "$out" | grep -m1 "tokens~")
  got="$(printf '%s' "$line" | grep -oE "bucket=[0-9]+" | cut -d= -f2) $(printf '%s' "$line" | grep -oE "regime=[A-Za-z0-9]+" | cut -d= -f2)"
  got="$got $(printf '%s' "$line" | grep -oE "gemm=[a-z]+\([a-z]+\)" | cut -d= -f2) $(printf '%s' "$line" | grep -oE "pairlogits_rows=[^ ]+" | cut -d= -f2)"
  got="$got $(printf '%s\n' "$out" | grep -q "WARNING: gemm=" && echo warn || echo -)"
  if [ "$got" = "$want" ]; then printf 'PASS  %-34s %s\n' "$name" "$got"
  else printf 'FAIL  %-34s got [%s] want [%s]\n' "$name" "$got" "$want"; fails=$((fails + 1)); [ -n "$line" ] || printf '%s\n' "$out" | head -5; fi
}

check "monomer"                     1  "512 A rocblas(auto) off(off) -"
check "8-mer"                       8  "3200 A rocblas(auto) off(off) -"
check "8-mer at 3,584 (A's top)"    8  "3584 A triton(auto) off(off) -"          -- --bucket 3584
check "10-mer (B, 3,968)"           10 "3968 B rocblas(auto) off(off) -"
check "10-mer at 4,096"             10 "4096 B rocblas(auto) 1024(auto) -"       -- --bucket 4096
check "11-mer (B, 4,352)"           11 "4352 B rocblas(auto) 1024(auto) -"
check "11-mer at 4,480 (B's top)"   11 "4480 B rocblas(auto) 1024(auto) -"       -- --bucket 4480
check "11-mer, --gemm triton"       11 "4352 B triton(given) off(off) -"         -- --gemm triton
check "11-mer, lever 0"             11 "4352 B rocblas(auto) off(zero) warn"     PERF_PAIRLOGITS_ROWS=0
check "11-mer, lever not a number"  11 "4352 B rocblas(auto) off(zero) warn"     PERF_PAIRLOGITS_ROWS=abc
check "11-mer, lever 2048"          11 "4352 B rocblas(auto) 2048(given) -"      PERF_PAIRLOGITS_ROWS=2048
check "11-mer, v12 cap (3,968)"     11 "4352 B triton(auto) off(off) -"          AF3JAX_ROCBLAS_B_MAX=3968
check "10-mer at 4,224, library"    10 "4224 B library(given) 1024(auto) -"      -- --bucket 4224 --gemm library
check "12-mer (C, 8 GCDs)"          12 "4736 C library(auto) off(off) -"
check "12-mer, lever set"           12 "4736 C library(auto) off(8gcd) -"        PERF_PAIRLOGITS_ROWS=1024
check "monomer, stock1"             1  "512 stock1 triton(auto) off(off) -"      -- --regime stock1

if [ "$fails" -eq 0 ]; then echo "all wrapper decisions as validated"; else echo "$fails decision(s) changed"; exit 1; fi
