#!/bin/bash
#SBATCH --job-name=opendde-validate
#SBATCH --partition=gpu-highmem
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=01:45:00
#SBATCH --output=opendde-validate.%j.out

# The opendde image against the native port, on one Setonix GPU node (8 MI250X GCDs).
#   sbatch --account=<project>-gpu validate.sbatch SIF=<the .sif> OUT=<dir> [NATIVE_D=<the native opendde-setonix tree>]
#          [WEIGHTS=<a complete weights root; default: fetched by part 1>] [REF600=<native 600-token P=2 output>]
#          [REF4096=<native 4,096-token P=8 output>] [PARTS=fetch,check,p2,p8]
#   (arguments, not variables: Setonix sets SBATCH_EXPORT=NONE). PARTS picks the sections, default all four.
#
# 1. fetch  `opendde fetch-weights` into node-local disk: the checkpoint and the four common/ files through the kit's own downloader
#           (upstream's URL table), each checked against the kit's pinned sha256, then compared with the native copies that survive.
#           This is the image's answer to the /scratch purge (which took two of the native common/ files on 2026-10-14). The parts
#           below use these weights unless WEIGHTS= is given.
# 2. check  `opendde check --patch-tests --predict`: versions, the kit's pin check, the patches' CPU tests, the weights against the pins,
#           every GPU library loaded from inside the image, a matmul on each of the 8 GCDs, and 600 tokens on one GCD under --det 1
#           through the stock line and through the patched line, each BYTE-IDENTICAL to the native port's reference structure
#           (tests/det600_stock.cif). That is the question the image has to answer: an interpreter built elsewhere (python-build-
#           standalone 3.12.12, not the OpenFold3 env's 3.12), a flat venv instead of the layered one, and torch's ROCm kernels pruned
#           to gfx90a must still produce the native bytes.
# 3. p2     600 tokens, the patched line row-sharded over 2 GCDs (shipped settings but --use_msa false --use_template false --sample 1),
#           against the native run of the same configuration (REF600, default $NATIVE_D/out/apb_a_600p2: predict_patched6.sh, which
#           at 600 tokens is predict_patched7.sh, the image's default). The sharded line is not bitwise run to run; natively two such
#           runs differed by 0.009-0.03 A. Fails above 0.10 A.
# 4. p8     4,096 tokens on all 8 GCDs, --cycle 1 --step 40 --use_msa false --use_template false --sample 1, against the native run
#           (REF4096, default $NATIVE_D/out/f4096_b: 1,262 s; two native runs differed by 0.236 A). Fails above 0.50 A; reports the
#           image's wall time against the native one (the container's own cost).
set -uo pipefail
for a in "$@"; do
  case "$a" in
    SIF=*|OUT=*|NATIVE_D=*|WEIGHTS=*|REF600=*|REF4096=*|PARTS=*) export "${a?}" ;;
    *) echo "unknown argument '$a': SIF= OUT= [NATIVE_D=] [WEIGHTS=] [REF600=] [REF4096=] [PARTS=]"; exit 2 ;;
  esac
done
PARTS=${PARTS:-fetch,check,p2,p8}
has() { case ",$PARTS," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
: "${SIF:?SIF=}" "${OUT:?OUT=}"
NATIVE_D=${NATIVE_D:-/scratch/pawsey1018/gbouras/opendde-setonix}
REF600=${REF600:-$NATIVE_D/out/apb_a_600p2}
REF4096=${REF4096:-$NATIVE_D/out/f4096_b}
O=$OUT/validate.${SLURM_JOB_ID:-local}
mkdir -p "$O" && cd "$O" || exit 3
[ -e "$SIF" ] || { echo "MISSING: $SIF"; exit 3; }
module load singularity/4.1.0-nohost > /dev/null 2>&1 || { echo "ABORT: no singularity/4.1.0-nohost module"; exit 3; }
L=/tmp/odde-validate-$USER-${SLURM_JOB_ID:-$$}
mkdir -p "$L"
export SINGULARITY_CACHEDIR=$L/sing-cache APPTAINER_CACHEDIR=$L/sing-cache
# The digest memo of the kit's weights check: here, not beside the native weights (this job writes nothing into the native tree).
export MODEL_OPT_WEIGHTS_DIGEST_DIR=$L/digest
C=(singularity exec --bind "$L:$L" --bind "$O:$O" "$SIF" opendde)    # never --rocm
FAIL=0
note() { printf '\n######## %s\n' "$*"; }
cif() { find "$1" -path '*/.rowpair' -prune -o -name '*_sample_0.cif' -print 2>/dev/null | head -1; }
rmsd() {   # rmsd <ref dir> <run dir> <limit>: CA RMSD (superposed) of the two sample-0 structures, FAIL above the limit
  local a b line r
  a=$(cif "$1"); b=$(cif "$2")
  if [ -z "$a" ] || [ -z "$b" ]; then echo "  structure missing (reference: ${a:-none}; image: ${b:-none})"; FAIL=1; return; fi
  line=$("${C[@]}" compare "$a" "$b" 2>&1 | tail -1)
  r=$(echo "$line" | grep -oE 'CA_RMSD_superposed=[0-9.]+' | cut -d= -f2)
  if awk -v r="${r:-99}" -v lim="$3" 'BEGIN { exit !(r <= lim) }'; then echo "  $line  (limit $3 A): OK"
  else echo "  $line  (limit $3 A): FAILED"; FAIL=1; fi
}

echo "=== node $(hostname)  job ${SLURM_JOB_ID:-?}  $(date)  image $SIF ($(du -h "$SIF" | cut -f1)) ==="
"${C[@]}" version | sed 's/^/  /'
t0=$SECONDS; cat "$SIF" > /dev/null; echo "pre-read the image: $((SECONDS - t0)) s"

# ---------------------------------------------------------------------------------------------------- 1. fetch
if has fetch; then
  note "1. opendde fetch-weights into node-local disk, then against the native copies"
  ODDE_LOCAL=$L/fetch "${C[@]}" fetch-weights "$L/weights" > "$O/fetch.log" 2>&1; rc=$?
  tail -8 "$O/fetch.log" | sed 's/^/  /'
  echo "  fetch rc=$rc"; [ "$rc" -eq 0 ] || FAIL=1
  for f in checkpoint/opendde.pt common/components.cif common/components.cif.rdkit_mol.pkl common/obsolete_to_successor.json \
           common/release_date_cache.json; do
    if [ ! -e "$NATIVE_D/weights/$f" ]; then echo "  $f: no native copy left to compare (the purge)"
    elif cmp -s "$L/weights/$f" "$NATIVE_D/weights/$f"; then echo "  $f = native"
    else echo "  $f DIFFERS from the native copy"; FAIL=1; fi
  done
  WEIGHTS=${WEIGHTS:-$L/weights}
fi
WEIGHTS=${WEIGHTS:-$NATIVE_D/weights}

# ---------------------------------------------------------------------------------------------------- 2. check
if has check; then
  note "2. opendde check --patch-tests --predict (all 8 GCDs; 600 tokens --det 1, stock and patched, against the native reference)"
  ODDE_LOCAL=$L/check "${C[@]}" check --weights "$WEIGHTS" --patch-tests --predict --keep "$O/check_predict" > check.log 2>&1; rc=$?
  sed 's/^/  /' check.log
  echo "check rc=$rc"; [ "$rc" -eq 0 ] || FAIL=1
fi

# ---------------------------------------------------------------------------------------------------- 3. p2
if has p2; then
  note "3. 600 tokens, patched line on 2 GCDs, against the native run ($REF600)"
  t0=$SECONDS
  ROCR_VISIBLE_DEVICES=0,1 ODDE_LOCAL=$L/p2 "${C[@]}" pred --weights "$WEIGHTS" --n_gpu 2 -i /opt/opendde/tests/q600.json -o "$O/p2" \
      --use_msa false --use_template false --sample 1 > "$O/p2.log" 2>&1; rc=$?
  echo "  rc=$rc  wall $((SECONDS - t0)) s  $(grep -a '^ODDE_RESULT' "$O/p2.log" | tail -1)"
  [ "$rc" -eq 0 ] || FAIL=1
  rmsd "$REF600" "$O/p2" 0.10
fi

# ---------------------------------------------------------------------------------------------------- 4. p8
if has p8; then
  note "4. 4,096 tokens, patched line on 8 GCDs, against the native run ($REF4096: 1,262 s)"
  t0=$SECONDS
  ODDE_LOCAL=$L/p8 "${C[@]}" pred --weights "$WEIGHTS" --n_gpu 8 -i /opt/opendde/tests/q4096.json -o "$O/p8" \
      --cycle 1 --step 40 --use_msa false --use_template false --sample 1 > "$O/p8.log" 2>&1; rc=$?
  echo "  rc=$rc  wall $((SECONDS - t0)) s (native 1,262 s)  $(grep -a '^ODDE_RESULT' "$O/p8.log" | tail -1)"
  [ "$rc" -eq 0 ] || FAIL=1
  rmsd "$REF4096" "$O/p8" 0.50
fi

note "VALIDATE $([ "$FAIL" -eq 0 ] && echo PASS || echo FAILED)  (job ${SLURM_JOB_ID:-local}, $(date +%T))"
echo "outputs in $O"
exit "$FAIL"
