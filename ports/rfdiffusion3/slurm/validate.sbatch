#!/bin/bash
#SBATCH --job-name=rfdiffusion3-validate
#SBATCH --partition=gpu-highmem
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=rfdiffusion3-validate.%j.out

# The rfdiffusion3 image against the native port, on one Setonix GPU node (8 MI250X GCDs). gpu-highmem, exclusive, because the design
# sections compare runs and a shared node makes them incomparable.
#   sbatch --account=<project>-gpu validate.sbatch SIF=<the .sif> OUT=<dir> NATIVE_W=<the native rfdiffusion-setonix tree> \
#          WEIGHTS=<dir holding rfd3/rfd3_latest.ckpt> [PARTS=check,designs,spread]
#   (arguments, not variables: Setonix sets SBATCH_EXPORT=NONE). PARTS picks the sections, default all three.
#
# 1. check    `rfdiffusion3 check --design`: both interpreters and the overlay's state, the kit's own pin check in each, every GPU
#             library loaded from inside the image, a matmul on each of the 8 GCDs, and -- the section that matters most on this card --
#             apex's FusedRMSNorm against a float64 reference. The unpatched kernel is 29 % wrong here and makes rubbish designs, so
#             this is the image proving which kernel it carries. `--design` then designs through `off` and `exact` inside the image and
#             compares them.
# 2. designs  the question the image has to answer: an interpreter built elsewhere (python-build-standalone 3.12.12, not cray-python
#             3.12.12) with ROCm libraries pruned to gfx90a must still produce the native port's designs. Four routes at one seed --
#             native stock, native exact, image stock, image exact -- compared design by design. The native tree's own
#             tools/rfd_compare.py does the comparing, so both halves are judged by the same code.
# 3. spread   stock against itself, twice in the image, on the same line. RFdiffusion3 does not always repeat itself here (the binder
#             example's `insulinr` key does not), so this measures the floor that section 2's numbers have to be read against. Without
#             it a difference in section 2 cannot be attributed.
set -uo pipefail
for a in "$@"; do
  case "$a" in
    SIF=*|OUT=*|NATIVE_W=*|WEIGHTS=*|PARTS=*) export "${a?}" ;;
    *) echo "unknown argument '$a': SIF= OUT= NATIVE_W= WEIGHTS= [PARTS=]"; exit 2 ;;
  esac
done
PARTS=${PARTS:-check,designs,spread}
has() { case ",$PARTS," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
: "${SIF:?SIF=}" "${OUT:?OUT=}" "${NATIVE_W:?NATIVE_W=}" "${WEIGHTS:?WEIGHTS=}"
mkdir -p "$OUT"
export RFD_LOCAL=${RFD_LOCAL:-/tmp/rfd-$USER-${SLURM_JOB_ID:-nojob}}
mkdir -p "$RFD_LOCAL"
module load singularity/4.1.0-nohost 2>/dev/null || true
C="singularity exec --bind $RFD_LOCAL:$RFD_LOCAL --bind $WEIGHTS:/weights --bind $OUT:$OUT $SIF rfdiffusion3"   # never --rocm
SPEC_IMG=/opt/rfdiffusion3/foundry-4010e3e2/models/rfd3/docs/examples/protein_binder_design.json
SPEC_NAT=$NATIVE_W/foundry-4010e3e2/models/rfd3/docs/examples/protein_binder_design.json
echo "node $(hostname) start $(date -Is)"
echo "SIF=$SIF  NATIVE_W=$NATIVE_W  WEIGHTS=$WEIGHTS  PARTS=$PARTS"

if has check; then
  echo "######## 1. check (with --design)"
  $C check --design 2>&1 | tail -60
  echo "check rc=${PIPESTATUS[0]}"
fi

if has designs; then
  echo "######## 2. the same line through four routes, compared design by design"
  L=(inputs=SPEC out_dir=DIR n_batches=1 diffusion_batch_size=8 seed=101)
  img() { ROCR_VISIBLE_DEVICES=$2 $C design --mode "$1" --det 1 inputs="$SPEC_IMG" out_dir="$OUT/img_$1" \
            n_batches=1 diffusion_batch_size=8 seed=101 > "$OUT/img_$1.log" 2>&1; echo "image $1 rc=$?"; }
  nat() { ( ROCR_VISIBLE_DEVICES=$2; export RFD_ROOT=$NATIVE_W
            . "$NATIVE_W/scripts/rfd-env.sh" kit > /dev/null 2>&1 || exit 97
            bash "$NATIVE_W/kit/rfdiffusion3/run.sh" design --config h100 --mode "$1" --det 1 \
                 inputs="$SPEC_NAT" out_dir="$OUT/nat_$1" n_batches=1 diffusion_batch_size=8 seed=101 \
          ) > "$OUT/nat_$1.log" 2>&1; echo "native $1 rc=$?"; }
  img off 0 & img exact 1 & nat off 2 & nat exact 3 & wait
  CMP=$NATIVE_W/tools/rfd_compare.py
  PY=$NATIVE_W/venv-kit/bin/python
  for pair in "nat_off img_off" "nat_exact img_exact" "nat_off nat_exact" "img_off img_exact" "nat_off img_exact"; do
    set -- $pair
    [ -d "$OUT/$1" ] && [ -d "$OUT/$2" ] || { echo "==== $1 vs $2: missing"; continue; }
    echo "==== $1 vs $2"
    "$PY" "$CMP" "$OUT/$1" "$OUT/$2" --quiet --json "$OUT/cmp_$1_$2.json" | tail -3
  done
fi

if has spread; then
  echo "######## 3. the floor: stock against itself, twice in the image"
  for i in a b; do
    ROCR_VISIBLE_DEVICES=$([ "$i" = a ] && echo 4 || echo 5) \
      $C design --mode off --det 1 inputs="$SPEC_IMG" out_dir="$OUT/floor_$i" \
          n_batches=1 diffusion_batch_size=8 seed=101 > "$OUT/floor_$i.log" 2>&1 &
  done
  wait
  "$NATIVE_W/venv-kit/bin/python" "$NATIVE_W/tools/rfd_compare.py" "$OUT/floor_a" "$OUT/floor_b" --quiet \
      --json "$OUT/cmp_floor.json" | tail -3
fi
echo "VALIDATE done $(date -Is)"
