#!/bin/bash
#SBATCH --job-name=openfold3-validate
#SBATCH --partition=gpu-highmem
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=openfold3-validate.%j.out

# The openfold3 image against the native port, on one Setonix GPU node (8 MI250X GCDs, the node the native runs had).
#   sbatch --account=<project>-gpu validate.sbatch SIF=<the .sif> OUT=<dir> NATIVE_D=<the native af3-setonix tree> \
#          [WEIGHTS=<default $OUT/weights>] [PARTS=fetch,check,bytes]
#   (arguments, not variables: Setonix sets SBATCH_EXPORT=NONE). PARTS picks the sections, default all three.
#
# The native tree is only read: its stored outputs, its MSAs and its checkpoint. Its environment is not run -- the /scratch purge has
# been taking it since 2026-10-14 -- so the image is held to the native port's own outputs from 2026-10-10, byte for byte.
# 1. fetch  `openfold3 fetch-weights WEIGHTS`: OpenFold3's own downloader, then the S3 object's size and CRC64NVME (a file already there
#           is checked, not fetched). Then compared with the native copy, if the purge has left it.
# 2. check  `openfold3 check --weights WEIGHTS`: versions, the sources against the native port's digests, the checkpoint, a matmul on
#           each of the 8 GCDs, every GPU library loaded from inside the image.
# 3. bytes  the image against the native port's runs of the accuracy benchmark (job 50560285, 2026-10-10; THROUGHPUT.md "Accuracy
#           against deposited structures"), the same queries with the same MSAs, inputs pinned (OF3_REF_POS_SEED=42 OF3_CONFORMER_SEED=42),
#           MSA-only, 5 samples, every sample's structure, aggregated confidences and full confidences compared byte for byte:
#             D9   9NX3 (726 tokens),   P=2, the port's defaults       against out/legit-50560285-9NX3-D
#             S9   9NX3,                P=2, --stock-settings          against out/legit-50560285-9NX3-S
#             D12  12ZN (1,880 tokens), P=4, the port's defaults       against out/legit-50560285-12ZN-D
#             D8V  8V9T (3,991 tokens), P=8, the port's defaults       against out/legit-50560285-8V9T-D
#           The question the image has to answer: an interpreter built elsewhere (python-build-standalone 3.12.12, not conda-forge's
#           3.12.14), a venv instead of the conda env, the sources rebuilt from public pins and torch's ROCm kernels pruned to gfx90a
#           must still give the native bytes, on 2, 4 and 8 GCDs and on both routes.
set -uo pipefail
for a in "$@"; do
  case "$a" in
    SIF=*|OUT=*|NATIVE_D=*|WEIGHTS=*|PARTS=*) export "${a?}" ;;
    *) echo "unknown argument '$a': SIF= OUT= NATIVE_D= [WEIGHTS=] [PARTS=]"; exit 2 ;;
  esac
done
PARTS=${PARTS:-fetch,check,bytes}
has() { case ",$PARTS," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
: "${SIF:?SIF=}" "${OUT:?OUT=}" "${NATIVE_D:?NATIVE_D=}"
WEIGHTS=${WEIGHTS:-$OUT/weights}
CKPT=of3-ob-2025-06-30-174k.pt
O=$OUT/validate.${SLURM_JOB_ID:-local}
mkdir -p "$O" "$WEIGHTS" && cd "$O" || exit 3
[ -e "$SIF" ] || { echo "MISSING: $SIF"; exit 3; }
module load singularity/4.1.0-nohost > /dev/null 2>&1 || { echo "ABORT: no singularity/4.1.0-nohost module"; exit 3; }
L=/tmp/of3-validate-$USER-${SLURM_JOB_ID:-$$}
mkdir -p "$L"
export SINGULARITY_CACHEDIR=$L/sing-cache APPTAINER_CACHEDIR=$L/sing-cache
C=(singularity exec --bind "$L:$L" --bind "$O:$O" --bind "$WEIGHTS:$WEIGHTS" "$SIF" openfold3)    # never --rocm
FAIL=0
note() { printf '\n######## %s\n' "$*"; }

echo "=== node $(hostname)  job ${SLURM_JOB_ID:-?}  $(date)  image $SIF ($(du -h "$SIF" | cut -f1)) ==="
"${C[@]}" version | sed 's/^/  /'
t0=$SECONDS; cat "$SIF" > /dev/null; echo "pre-read the image: $((SECONDS - t0)) s"

# ---------------------------------------------------------------------------------------------------- 1. fetch
if has fetch; then
note "1. openfold3 fetch-weights $WEIGHTS (OpenFold3's own downloader; the S3 object's size and CRC64NVME), then against the native copy"
t0=$SECONDS
OF3_LOCAL=$L/fetch "${C[@]}" fetch-weights "$WEIGHTS" > "$O/fetch.log" 2>&1; rc=$?
grep -a -E '^\[weights\]' "$O/fetch.log" | sed 's/^/  /' | cut -c1-220
echo "fetch rc=$rc wall_s=$((SECONDS - t0))"; [ "$rc" -eq 0 ] || FAIL=1
if [ ! -e "$NATIVE_D/openfold3-cache/$CKPT" ]; then echo "  $CKPT: no native copy left (the /scratch purge)"
elif cmp -s "$WEIGHTS/$CKPT" "$NATIVE_D/openfold3-cache/$CKPT"; then echo "  $CKPT = the native copy, byte for byte"
else echo "  $CKPT DIFFERS from the native copy (or is missing here)"; FAIL=1; fi
fi
[ -s "$WEIGHTS/$CKPT" ] || { note "VALIDATE FAILED: $WEIGHTS/$CKPT is missing, so nothing can run (PARTS=fetch fetches it)"; exit 1; }

# ---------------------------------------------------------------------------------------------------- 2. check
if has check; then
note "2. openfold3 check --weights $WEIGHTS (all 8 GCDs)"
OF3_LOCAL=$L/check "${C[@]}" check --weights "$WEIGHTS" > check.log 2>&1; rc=$?
sed 's/^/  /' check.log
echo "check rc=$rc"; [ "$rc" -eq 0 ] || FAIL=1
fi

# ---------------------------------------------------------------------------------------------------- 3. bytes
if has bytes; then
note "3. the image against the native port's benchmark runs (job 50560285), byte for byte"
run_one() {   # tag id P route
  local tag=$1 id=$2 p=$3 route=$4 ref d q rc t0 extra=""
  ref=$NATIVE_D/out/legit-50560285-$id-$route
  q=$NATIVE_D/msacache/legit_$id/query_msa.json
  d=$O/bytes/$tag
  mkdir -p "$d"
  [ "$route" = S ] && extra=--stock-settings
  if [ ! -d "$ref" ] || [ ! -s "$q" ]; then
    printf '  %-4s MISSING: native outputs %s (%s) or query %s (%s)\n' "$tag" "$ref" "$([ -d "$ref" ] && echo ok || echo absent)" \
      "$q" "$([ -s "$q" ] && echo ok || echo absent)"
    FAIL=1; return
  fi
  t0=$SECONDS
  OF3_LOCAL=$L/$tag OF3_REF_POS_SEED=42 OF3_CONFORMER_SEED=42 \
    "${C[@]}" predict --weights "$WEIGHTS" -i "$q" -o "$d/out" -P "$p" $extra > "$d/run.log" 2>&1
  rc=$?
  printf '  %-4s %s P=%s %-8s rc=%s wall_s=%s  %s\n' "$tag" "$id" "$p" "$route" "$rc" "$((SECONDS - t0))" \
    "$(grep -a -m1 -oE 'runtime_s: [0-9.]+' "$d/run.log")"
  grep -a -h -E "^\[launch\] settings" "$d/run.log" | cut -c1-260 | sed 's/^/      /'
  if [ "$rc" -ne 0 ]; then
    FAIL=1; grep -a -E "Traceback|Error|refused|RESULT" "$d/run.log" "$d"/out/ranks/rank0.log 2>/dev/null | tail -6 | sed 's/^/      /'
    return
  fi
  if "${C[@]}" compare "$ref" "$d/out" --identical > "$d/compare.txt" 2>&1; then v=IDENTICAL; else v=DIFFER; FAIL=1; fi
  printf '  %-4s %s  %s\n' "$tag" "$v" "$(grep -a 'byte-identical' "$d/compare.txt" | tail -1 | sed -E 's/^ *//')"
  grep -a -E "seed [0-9]+ sample" "$d/compare.txt" | sed 's/^ */      /' | cut -c1-220
}
run_one D9  9NX3 2 D
run_one S9  9NX3 2 S
run_one D12 12ZN 4 D
run_one D8V 8V9T 8 D
fi

note "VALIDATE $([ "$FAIL" -eq 0 ] && echo PASS || echo FAILED)  (job ${SLURM_JOB_ID:-local}, $(date +%T))"
echo "outputs in $O; weights in $WEIGHTS"
exit "$FAIL"
