#!/bin/bash
# Dry-run tests of scripts/ptx-pred.sh's decisions (no GPU, no torch): the route, the command line and the environment each route hands
# the model process. A stub `python` first on PATH prints its argv and the recipe's variables instead of running anything; the weights
# check (python -I -S -) goes to the real python3. Run by .github/workflows/checks.yml; locally: bash ports/protenix/tests/test_launcher.sh
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PORT=$(cd "$HERE/.." && pwd)
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT

# a minimal kit tree (PINS.json names the seven files) and a complete weights root
mkdir -p "$T/kit/stock" "$T/w/checkpoint" "$T/w/common" "$T/bin"
cat > "$T/kit/stock/PINS.json" <<'EOF'
{"checkpoint": {"file": "checkpoint/protenix-v2.pt", "sha256": "x"},
 "data_caches": {"files": {"common/components.cif": {}, "common/components.cif.rdkit_mol.pkl": {}, "common/obsolete_release_date.csv": {},
                           "common/clusters-by-entity-40.txt": {}, "common/release_date_cache.json": {}, "common/obsolete_to_successor.json": {}}}}
EOF
for f in checkpoint/protenix-v2.pt common/components.cif common/components.cif.rdkit_mol.pkl common/obsolete_release_date.csv \
         common/clusters-by-entity-40.txt common/release_date_cache.json common/obsolete_to_successor.json; do : > "$T/w/$f"; done
cat > "$T/bin/python" <<'EOF'
#!/bin/bash
[ "${1:-}" = -I ] && exec python3 "$@"
echo "ARGV $*"
echo "ENV LEVERS=${MODEL_OPT_LEVERS_OFF:-} ALLOC=${PYTORCH_CUDA_ALLOC_CONF:-} CAP=${PXS_TRIATT_CHUNK_CAP:-} OPT=${PROTENIX_OPT-unset}"
EOF
chmod +x "$T/bin/python"

# the environment ptx-env.sh leaves (MODEL_OPT set, so ptx-pred.sh does not source it again)
base_env=(env -i PATH="$T/bin:$PATH" HOME="$T" MODEL_OPT="$T/kit" PROTENIX_ROOT_DIR="$T/w" PROTENIX_OPT=big
          MODEL_OPT_LEVERS_OFF=fastln_prebuilt,ln_core,sampler_fuse ROWPAIR_TRIATT_CORE=flash_triattn ROWPAIR_DIT_ROWS=engine)
PASS=0 FAIL=0
run() { "${base_env[@]}" "$@" bash "$PORT/scripts/ptx-pred.sh" ${ARGS[@]+"${ARGS[@]}"} 2> "$T/err"; }
expect() {   # expect <name> <want rc> <grep -E pattern on stdout> [more patterns...]   (a pattern starting with ! must NOT match)
  local name=$1 want=$2 out rc; shift 2
  out=$(run ${PRE[@]+"${PRE[@]}"}); rc=$?   # (the +-form: bash 3.2 treats an empty array as unset)
  local ok=1
  [ "$rc" = "$want" ] || ok=0
  for p in "$@"; do
    if [ "${p#!}" != "$p" ]; then printf '%s\n' "$out" | grep -qE -- "${p#!}" && ok=0
    else printf '%s\n' "$out" | grep -qE -- "$p" || ok=0; fi
  done
  if [ $ok = 1 ]; then PASS=$((PASS + 1)); echo "ok   $name"
  else FAIL=$((FAIL + 1)); echo "FAIL $name (rc $rc, want $want)"; printf '%s\n' "$out" | sed 's/^/     /'; sed 's/^/     stderr: /' "$T/err"; fi
}

PRE=(); ARGS=(-i q.json -o out)
expect "one GCD: big, transition_core off, expandable segments off, stock's chunk" 0 \
  '^ARGV -m protenix_opt pred --mode big -i q.json -o out --model_name protenix-v2 --triatt_kernel torch --trimul_kernel torch$' \
  'LEVERS=fastln_prebuilt,ln_core,sampler_fuse,transition_core ' 'ALLOC=expandable_segments:False ' 'CAP= '

ARGS=(--n_gpu 8 -i q.json -o out --seeds 101)
expect "--n_gpu 8: the sharded line, chunk cap 4, the kit's allocator" 0 \
  '^ARGV -m protenix_opt pred --mode big --n_gpu 8 -i q.json -o out --seeds 101 --model_name protenix-v2' 'ALLOC= ' 'CAP=4 '

ARGS=(--n_gpu=2 -i q.json -o out)
expect "--n_gpu=2 (the = form)" 0 '--mode big --n_gpu 2 ' 'CAP=4 '

ARGS=(--mode stock -i q.json -o out)
expect "stock: stock's own CLI, PROTENIX_OPT unset, no --mode passed on" 0 \
  '^ARGV -c .*protenix_cli\(\) pred -i q.json -o out --model_name protenix-v2 --triatt_kernel torch --trimul_kernel torch$' 'OPT=unset' 'LEVERS= ' '!--mode[ =]'

ARGS=(--mode stock --n_gpu 2 -i q.json -o out)
expect "stock refuses --n_gpu 2" 2

ARGS=(--mode off --n_gpu 2 -i q.json -o out)
expect "off: the kit's stock route, --n_gpu passed on for the kit to refuse" 0 '^ARGV -m protenix_opt pred --mode off --n_gpu 2 ' 'LEVERS= '

ARGS=(--mode=off -i q.json -o out)
expect "--mode=off (the = form), no --n_gpu" 0 '^ARGV -m protenix_opt pred --mode off -i q.json' '!--n_gpu'

ARGS=(-i q.json -o out --model_name protenix_base_default_v1.0.0 --triatt_kernel=torch)
expect "given --model_name / --triatt_kernel= are kept, not doubled" 0 '--model_name protenix_base_default_v1.0.0' \
  '!--model_name protenix-v2' '--triatt_kernel=torch' '!--triatt_kernel torch' '--trimul_kernel torch'

ARGS=(--mode exact -i q.json -o out)
expect "exact is refused (not validated on gfx90a)" 2
ARGS=(--mode fast -i q.json -o out)
expect "fast is refused (not validated on gfx90a)" 2
ARGS=(--mode turbo -i q.json -o out)
expect "an unknown mode is refused" 2
ARGS=(--n_gpu 0 -i q.json -o out)
expect "--n_gpu 0 is refused" 2
ARGS=(--n_gpu x -i q.json -o out)
expect "--n_gpu x is refused" 2

PRE=(MODEL_OPT_LEVERS_OFF=fastln_prebuilt,transition_core,ln_core,sampler_fuse PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)
ARGS=(-i q.json -o out)
expect "a set lever list and allocator setting win; transition_core not doubled" 0 \
  'LEVERS=fastln_prebuilt,transition_core,ln_core,sampler_fuse ' 'ALLOC=expandable_segments:True '

PRE=(PXS_TRIATT_CHUNK_CAP=2)
ARGS=(--n_gpu 8 -i q.json -o out)
expect "a set chunk cap wins" 0 'CAP=2 '

rm -f "$T/w/common/release_date_cache.json"
PRE=(); ARGS=(-i q.json -o out)
expect "a missing weight file stops the run before Python starts" 4 '!^ARGV'
grep -q "missing common/release_date_cache.json" "$T/err" && { PASS=$((PASS + 1)); echo "ok   ... and names the file"; } \
  || { FAIL=$((FAIL + 1)); echo "FAIL the missing file is not named:"; cat "$T/err"; }

PRE=(PROTENIX_ROOT_DIR=)
expect "no weights root at all stops the run" 4 '!^ARGV'

echo "$PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
