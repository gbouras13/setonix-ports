#!/bin/bash
# test_entry.sh -- the entry point's predict decisions, through its --dry-run: no GPU, no weights (a one-byte stand-in), no model.
# The entry point is the image's (/usr/local/bin/openfold3, on /opt/openfold3), so the image build runs this (container/Dockerfile, step 5).
#   P auto: 2 GCDs to 1,600 residues, 4 to 3,100, else 8; an explicit -P wins; -P must be 2, 4, 8 or auto
#   the runner yaml: the 128-row lazytempl yaml, or its 256-row twin with --stock-settings; templates and the MSA server off unless asked
#   a protein chain without an MSA is refused unless --no-msa or --msa-server; no weights, or a checkpoint missing, is refused
set -uo pipefail
E=${OF3_ENTRY:-openfold3}
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/w" "$T/msa"
printf x > "$T/w/of3-ob-2025-06-30-174k.pt"
printf x > "$T/msa/a.npz"
query() {   # file copies length [msa]
  local seq; seq=$(head -c "$3" /dev/zero | tr '\0' 'A')
  local ids; ids=$(python3 -c "import json,sys; print(json.dumps([chr(65 + i) for i in range(int(sys.argv[1]))]))" "$2")
  local msa=null; [ "${4:-1}" = 1 ] && msa="[\"$T/msa/a.npz\"]"
  printf '{"queries": {"t": {"chains": [{"molecule_type": "protein", "chain_ids": %s, "sequence": "%s", "main_msa_file_paths": %s}]}}}\n' \
    "$ids" "$seq" "$msa" > "$1"
}
query "$T/q600.json" 3 200
query "$T/q1600.json" 4 400
query "$T/q1601.json" 1 1601
query "$T/q3100.json" 2 1550
query "$T/q3101.json" 1 3101
query "$T/qnomsa.json" 2 100 0
PASS=0 FAILN=0
expect() {   # name want-rc grep-pattern args...
  local name=$1 want=$2 pat=$3; shift 3
  local out rc
  out=$(OF3_LOCAL="$T/local" "$E" predict "$@" --dry-run 2>&1); rc=$?
  if [ "$rc" = "$want" ] && printf '%s\n' "$out" | grep -q -E -- "$pat"; then
    PASS=$((PASS + 1)); echo "  ok    $name"
  else
    FAILN=$((FAILN + 1)); echo "  FAIL  $name (rc=$rc, want $want, pattern '$pat')"; printf '%s\n' "$out" | sed 's/^/        /' | tail -5
  fi
}
W="--weights $T/w"
# shellcheck disable=SC2086
{
expect "600 residues -> P=2"          0 "^PLAN P=2 RUNNER=.*sharded-lazytempl\.yml STOCK=0 USE_TEMPLATES=false USE_MSA_SERVER=false RES=600$" $W -i "$T/q600.json" -o "$T/o"
expect "1,600 -> P=2"                 0 "^PLAN P=2 .*RES=1600$"  $W -i "$T/q1600.json" -o "$T/o"
expect "1,601 -> P=4"                 0 "^PLAN P=4 .*RES=1601$"  $W -i "$T/q1601.json" -o "$T/o"
expect "3,100 -> P=4"                 0 "^PLAN P=4 .*RES=3100$"  $W -i "$T/q3100.json" -o "$T/o"
expect "3,101 -> P=8"                 0 "^PLAN P=8 .*RES=3101$"  $W -i "$T/q3101.json" -o "$T/o"
expect "-P 8 wins"                    0 "^PLAN P=8 .*RES=600$"   $W -i "$T/q600.json" -o "$T/o" -P 8
expect "--stock-settings: c256 yaml"  0 "^PLAN P=2 RUNNER=.*sharded-lazytempl-c256\.yml STOCK=1 " $W -i "$T/q600.json" -o "$T/o" --stock-settings
expect "--templates"                  0 "USE_TEMPLATES=true USE_MSA_SERVER=false" $W -i "$T/q600.json" -o "$T/o" --templates
expect "no MSA: refused"              2 "carry no MSA"           $W -i "$T/qnomsa.json" -o "$T/o"
expect "no MSA, --no-msa: runs"       0 "^PLAN P=2 .*RES=200$"   $W -i "$T/qnomsa.json" -o "$T/o" --no-msa
expect "no MSA, --msa-server: runs"   0 "USE_MSA_SERVER=true"    $W -i "$T/qnomsa.json" -o "$T/o" --msa-server
expect "-P 3: refused"                2 "must be 2, 4, 8 or auto" $W -i "$T/q600.json" -o "$T/o" -P 3
expect "no weights: refused"          2 "no weights"             -i "$T/q600.json" -o "$T/o"
expect "no checkpoint: refused"       2 "no of3-ob-2025-06-30-174k.pt" --weights "$T/msa" -i "$T/q600.json" -o "$T/o"
expect "no -o: refused"               2 "give the output directory" $W -i "$T/q600.json"
}
echo "test_entry: $PASS passed, $FAILN failed"
[ "$FAILN" -eq 0 ]
