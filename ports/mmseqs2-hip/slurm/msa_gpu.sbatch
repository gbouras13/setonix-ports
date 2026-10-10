#!/bin/bash --login
#SBATCH --job-name=msa-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8,tmp:1000G
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=msa-gpu.%j.out
#
# ColabFold MSAs on one Setonix GPU node (8 MI250X GCDs) with the HIP build of MMseqs2 (see BUILD.md).
#
#   sbatch --account=<project>-gpu msa_gpu.sbatch [MMSEQS=... MMSEQS_WRAP=... CFSEARCH=... COLABFOLD_DB=... THREADS=... STAGE=...] \
#          QUERIES.fasta OUTDIR [extra colabfold_search arguments]
# The settings come first, as KEY=VALUE arguments (Setonix sets SBATCH_EXPORT=NONE, so variables exported before sbatch never reach
# the job); exported variables still work when this script is run with bash inside a job. The ColabFold image's `colabfold search`
# does the same with its own binary.
#
# QUERIES.fasta: one record per prediction. A complex is one record with its chains joined by ':' (ColabFold's
# convention); colabfold_search then also writes paired MSAs. Extra arguments go straight to colabfold_search, e.g.
# --use-templates 1 (pdb100 hits), --af3-json (AlphaFold3-style input JSON per query), --pair-mode paired.
#
# By default (STAGE=1) the databases are first copied to the node's NVMe (/tmp; 522 GB in ~4.5 min), and the
# database-reading steps run with --db-load-mode 2 through mmseqs-dbload2. That was 2.1x faster than reading Lustre
# directly for 246 phage proteins, with byte-identical MSAs (job 50235377). STAGE=0 reads straight from Lustre,
# which is only worth it for a handful of queries.
#
# Settings:
#   MMSEQS        the HIP mmseqs binary (required)
#   MMSEQS_WRAP   mmseqs-dbload2 (default: next to MMSEQS; scripts/mmseqs-dbload2 in this repository)
#   CFSEARCH      colabfold_search (pip install colabfold==1.6.3)
#   COLABFOLD_DB  Pawsey's ColabFold database root (UniRef30 2302, envDB 202108, pdb100; already GPU-padded)
#   THREADS       CPU threads for the non-prefilter steps (default 64 = the node's cores)
set -euo pipefail
while [ $# -gt 0 ]; do
  case "$1" in
    MMSEQS=*|MMSEQS_WRAP=*|CFSEARCH=*|COLABFOLD_DB=*|THREADS=*|STAGE=*) export "${1?}"; shift ;;
    *) break ;;
  esac
done
Q=$(readlink -f "${1:?usage: sbatch msa_gpu.sbatch QUERIES.fasta OUTDIR [colabfold_search args]}")
OUT=${2:?usage: sbatch msa_gpu.sbatch QUERIES.fasta OUTDIR [colabfold_search args]}
shift 2
MMSEQS=${MMSEQS:?MMSEQS=<the HIP mmseqs binary>}
MMSEQS_WRAP=${MMSEQS_WRAP:-$(dirname "$MMSEQS")/mmseqs-dbload2}
CFSEARCH=${CFSEARCH:-colabfold_search}
DB=${COLABFOLD_DB:-/scratch/references/colabfold_jun2026/database}
THREADS=${THREADS:-64}
STAGE=${STAGE:-1}
U=uniref30_2302_db
E=colabfold_envdb_202108_db
P=pdb100_230517
SUB=(colabfold_uniref30 colabfold_envdb pdb100)      # their subdirectories in Pawsey's layout
mkdir -p "$OUT"
echo "node $(hostname)  $(date -Is)  mmseqs $("$MMSEQS" version)  queries $(grep -c '^>' "$Q")  out $OUT  stage $STAGE"

# colabfold_search needs bare database names: it writes template output to OUT/<--db2> and OUT/<name>_<--db2>.m8.
# So all three databases go into one flat directory, as copies on the node's NVMe (STAGE=1) or as symlinks.
BASE=${TMPDIR:-/tmp}/cfdb-${SLURM_JOB_ID:-$$}
mkdir -p "$BASE"
FILES=$(ls -d "$DB/${SUB[0]}/$U"* "$DB/${SUB[1]}/$E"* "$DB/${SUB[2]}/$P"*)
MM=$MMSEQS
if [ "$STAGE" = 1 ]; then
    t0=$(date +%s)
    # every file of the three databases, in 4 GiB chunks, 16 readers, read with O_DIRECT from Lustre
    for f in $FILES; do
        dst=$BASE/$(basename "$f"); size=$(stat -L -c %s "$f")
        truncate -s "$size" "$dst"
        n=$(( (size + (4 << 30) - 1) / (4 << 30) ))
        for ((i = 0; i < n; i++)); do echo "$f $dst $i"; done
    done | xargs -P 16 -n 3 sh -c 'dd if="$0" of="$1" bs=64M skip=$(( $2 * 64 )) seek=$(( $2 * 64 )) count=64 iflag=direct conv=notrunc status=none'
    sync
    echo "staged $(du -s --block-size=1G "$BASE" | cut -f1) GB to $BASE in $(( $(date +%s) - t0 )) s"
    export MMSEQS_REAL=$MMSEQS; MM=$MMSEQS_WRAP
else
    for f in $FILES; do ln -sfn "$f" "$BASE/$(basename "$f")"; done
fi

# One resident GPU server per database: every colabfold_search iteration then reuses the database already on the
# GCDs. --max-seqs must match colabfold_search's (10000).
SRV=()
cleanup () { [ ${#SRV[@]} -gt 0 ] && kill "${SRV[@]}" 2>/dev/null; wait 2>/dev/null || true; }
trap cleanup EXIT
for d in "$U" "$E"; do
    "$MMSEQS" gpuserver "$BASE/$d" --max-seqs 10000 --db-load-mode 0 --prefilter-mode 1 > "$OUT/gpuserver.$(basename "$d").log" 2>&1 &
    SRV+=($!)
done

t0=$(date +%s)
"$CFSEARCH" --mmseqs "$MM" --gpu 1 --gpu-server 1 --threads "$THREADS" \
    --db1 "$U" --db2 "$P" --db3 "$E" --use-env 1 "$@" \
    "$Q" "$BASE" "$OUT"
echo "colabfold_search: $(( $(date +%s) - t0 )) s, $(ls "$OUT"/*.a3m 2>/dev/null | wc -l) a3m files in $OUT"
