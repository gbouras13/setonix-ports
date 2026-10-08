#!/bin/bash
#SBATCH --job-name=af3jax-pull
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=af3jax-pull.%j.out

# Pull the af3_jax image from ghcr.io and turn it into a SIF, on a compute node (Pawsey asks for pulls there, never on a login node).
#   sbatch --account=<project> pull_image.sbatch [IMAGE] [DEST_DIR]
#     IMAGE     default ghcr.io/gbouras13/setonix-ports/af3jax:main; a tag (:3.1.4-1) or a digest (@sha256:...) from the Actions summary
#     DEST_DIR  default $MYSCRATCH/singularity. On /scratch the SIF is purged 21 days after it was last read; it is one file, so
#               rerunning this job brings it back. $MYSOFTWARE/singularity is not purged, if your quota has room (~5 GB). $MYSCRATCH
#               follows your DEFAULT project: if the image belongs elsewhere, give the directory (/scratch/<project>/$USER/singularity).
# Then: the image's provenance and its self-check without a GPU (versions, patches, the AlphaFold 3 import, every ROCm library resolved
# inside the image). The GPU check runs in a GPU job: singularity exec "$SIF" af3jax check
set -euo pipefail
IMAGE=${1:-ghcr.io/gbouras13/setonix-ports/af3jax:main}
DEST=${2:-${MYSCRATCH:?MYSCRATCH is not set}/singularity}
case "$IMAGE" in docker://*) IMAGE=${IMAGE#docker://};; esac
name=$(basename "${IMAGE%%@*}" | tr ':' '_')
case "$IMAGE" in *@sha256:*) d=${IMAGE##*@sha256:}; name="${name}_${d:0:12}";; esac
SIF=$DEST/$name.sif
mkdir -p "$DEST"
echo "=== $(hostname) job ${SLURM_JOB_ID:-?} $(date)  image docker://$IMAGE -> $SIF ==="

# Layers and the unpacked root filesystem (~25 GB, ~60k files) go to node-local /tmp, never to the 1 GB $HOME (Singularity's default
# cache) and never to /scratch (its inode quota). If /tmp is memory-backed on this node type, --mem above covers it.
T=/tmp/af3jax-pull.${SLURM_JOB_ID:-$$}
mkdir -p "$T/cache" "$T/tmp"
trap 'rm -rf "$T"' EXIT
echo "node-local /tmp: $(df -h /tmp | awk 'NR==2 {print $4 " free of " $2 " (" $1 ")"}')"

module load singularity/4.1.0-nohost > /dev/null 2>&1 || { echo "ABORT: no singularity/4.1.0-nohost module"; exit 3; }
# after the module, which sets its own cache directory under $MYSOFTWARE
export SINGULARITY_CACHEDIR=$T/cache SINGULARITY_TMPDIR=$T/tmp APPTAINER_CACHEDIR=$T/cache APPTAINER_TMPDIR=$T/tmp
echo "singularity: $(singularity --version)"
t0=$SECONDS
singularity pull --force "$SIF.partial" "docker://$IMAGE"
mv -f "$SIF.partial" "$SIF"
echo "pulled in $((SECONDS - t0)) s: $SIF ($(du -h "$SIF" | cut -f1))"

echo; echo "=== provenance ==="
singularity inspect --labels "$SIF" | grep -E "org.opencontainers.image.(source|revision|version|created)" || true
singularity exec "$SIF" af3jax version

echo; echo "=== self-check without a GPU ==="
singularity exec "$SIF" af3jax check --no-gpu
echo; echo "SIF ready: $SIF"
echo "next, in a GPU job:  singularity exec $SIF af3jax check     (no --rocm)"
