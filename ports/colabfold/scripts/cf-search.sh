#!/bin/bash
# cf-search.sh -- ColabFold MSAs: colabfold_search on the GPU build of MMseqs2 (ports/mmseqs2-hip), one Setonix GPU node.
#   cf-search.sh QUERIES.fasta OUTDIR [colabfold_search options...]          (in the image: colabfold search QUERIES.fasta OUTDIR ...)
# QUERIES.fasta: one record per prediction; a complex is one record with its chains joined by ':' (colabfold_search then also writes
# paired MSAs). Options go straight to colabfold_search, e.g. --use-templates 1 (pdb100 hits), --af3-json, --pair-mode paired.
#
# af3-setonix/mmseqs2-hip/msa_gpu.sbatch made into a command (its run: job 50421759, 8 long chains with templates, 752 s):
#   - the three databases (UniRef30 2302, envDB 202108, pdb100) go into one flat directory under bare names, as colabfold_search's
#     template step needs (it writes OUT/<--db2> and OUT/<name>_<--db2>.m8);
#   - STAGE=1 copies them to the node's NVMe first (/tmp: book it with --gres=gpu:8,tmp:1000G; 522 GB in ~4.5 min) and runs the
#     database-reading steps with --db-load-mode 2 (mmseqs-dbload2): 2.1x faster than Lustre for 246 phage proteins, with byte-identical
#     MSAs (job 50235377). STAGE=0 reads Lustre directly, which is only worth it for a handful of queries;
#   - one gpuserver per database (UniRef30, envDB), so every search iteration finds its database resident on the GCDs (--max-seqs as
#     colabfold_search's 10000);
#   - colabfold_search --gpu 1 --gpu-server 1 --db1 uniref30_2302_db --db2 pdb100_230517 --db3 colabfold_envdb_202108_db --use-env 1.
# GPU=0 runs MMseqs2's CPU search instead (the same binary, no servers).
#
# Environment (all optional): COLABFOLD_DB (Pawsey's database root; already GPU-padded), STAGE (1 | 0; default 1 when the temporary
# directory has room for the three databases, else 0), THREADS (default: the CPUs this process may use), GPU (1 | 0), MMSEQS (the GPU
# MMseqs2), CFSEARCH (colabfold_search). Setonix starts jobs from a fresh login environment (SBATCH_EXPORT=NONE): set them in the job.
set -euo pipefail
W=${CFS_W:-$(cd "$(dirname "$0")/.." && pwd)}
USAGE="usage: cf-search.sh QUERIES.fasta OUTDIR [colabfold_search options]"
Q=$(readlink -f "${1:?$USAGE}")
OUT=${2:?$USAGE}
shift 2
MMSEQS=${MMSEQS:-$W/bin/mmseqs}
WRAP=${MMSEQS_WRAP:-$W/bin/mmseqs-dbload2}
CFSEARCH=${CFSEARCH:-$W/venv/bin/colabfold_search}
DB=${COLABFOLD_DB:-/scratch/references/colabfold_jun2026/database}
THREADS=${THREADS:-$(nproc)}
GPU=${GPU:-1}
U=uniref30_2302_db
E=colabfold_envdb_202108_db
P=pdb100_230517
SUB=(colabfold_uniref30 colabfold_envdb pdb100)      # their subdirectories in Pawsey's layout
for f in "$Q" "$MMSEQS" "$CFSEARCH"; do [ -e "$f" ] || { echo "cf-search: MISSING $f" >&2; exit 3; }; done
mkdir -p "$OUT"
mapfile -t FILES < <(ls -d "$DB/${SUB[0]}/$U"* "$DB/${SUB[1]}/$E"* "$DB/${SUB[2]}/$P"* 2>/dev/null)
[ "${#FILES[@]}" -gt 0 ] || { echo "cf-search: no databases under $DB (${SUB[*]})" >&2; exit 3; }
TMP=${TMPDIR:-/tmp}
if [ -z "${STAGE:-}" ]; then
  need=$(du -c -L --block-size=1G "${FILES[@]}" | tail -1 | cut -f1)
  free=$(df --output=avail --block-size=1G "$TMP" | tail -1 | tr -dc 0-9)
  if [ "${free:-0}" -gt $((need + 20)) ]; then STAGE=1; else STAGE=0; fi
  echo "cf-search: databases ${need} GB, ${free:-?} GB free in $TMP -> STAGE=$STAGE" >&2
fi
echo "cf-search: node $(hostname)  $(date -Is)  mmseqs $("$MMSEQS" version)  queries $(grep -c '^>' "$Q")  out $OUT  stage $STAGE  gpu $GPU  threads $THREADS"

BASE=$TMP/cfdb-${SLURM_JOB_ID:-$$}
mkdir -p "$BASE"
MM=$MMSEQS
if [ "$STAGE" = 1 ]; then
  t0=$(date +%s)
  # every file of the three databases in 4 GiB chunks, 16 readers, read with O_DIRECT from Lustre; a file already staged under this job's
  # directory with the right size (an earlier search in the same job, CFS_KEEP_STAGED=1) is not copied again
  for f in "${FILES[@]}"; do
    dst=$BASE/$(basename "$f"); size=$(stat -L -c %s "$f")
    [ -f "$dst" ] && [ ! -L "$dst" ] && [ "$(stat -c %s "$dst")" = "$size" ] && [ -f "$BASE/.staged" ] && continue
    truncate -s "$size" "$dst"
    n=$(( (size + (4 << 30) - 1) / (4 << 30) ))
    for ((i = 0; i < n; i++)); do echo "$f $dst $i"; done
  done | xargs -P 16 -n 3 sh -c 'dd if="$0" of="$1" bs=64M skip=$(( $2 * 64 )) seek=$(( $2 * 64 )) count=64 iflag=direct conv=notrunc status=none'
  sync
  touch "$BASE/.staged"
  echo "cf-search: staged $(du -s --block-size=1G "$BASE" | cut -f1) GB to $BASE in $(( $(date +%s) - t0 )) s"
  export MMSEQS_REAL=$MMSEQS
  MM=$WRAP
else
  for f in "${FILES[@]}"; do [ -f "$BASE/$(basename "$f")" ] && [ ! -L "$BASE/$(basename "$f")" ] || ln -sfn "$f" "$BASE/$(basename "$f")"; done
fi

SRV=()
cleanup () {
  [ ${#SRV[@]} -gt 0 ] && kill "${SRV[@]}" 2> /dev/null
  wait 2> /dev/null || true
  [ "${CFS_KEEP_STAGED:-0}" = 1 ] || rm -rf "$BASE"
}
trap cleanup EXIT
GPUARGS=()
if [ "$GPU" = 1 ]; then
  # one resident GPU server per database; --max-seqs must match colabfold_search's (10000)
  for d in "$U" "$E"; do
    "$MMSEQS" gpuserver "$BASE/$d" --max-seqs 10000 --db-load-mode 0 --prefilter-mode 1 > "$OUT/gpuserver.$d.log" 2>&1 &
    SRV+=($!)
  done
  GPUARGS=(--gpu 1 --gpu-server 1)
fi

t0=$(date +%s)
"$CFSEARCH" --mmseqs "$MM" "${GPUARGS[@]}" --threads "$THREADS" \
  --db1 "$U" --db2 "$P" --db3 "$E" --use-env 1 "$@" \
  "$Q" "$BASE" "$OUT"
echo "cf-search: colabfold_search $(( $(date +%s) - t0 )) s, $(find "$OUT" -maxdepth 1 -name '*.a3m' | wc -l) a3m files in $OUT"
