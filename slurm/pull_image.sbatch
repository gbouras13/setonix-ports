#!/bin/bash
#SBATCH --job-name=image-pull
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=image-pull.%j.out

# Pull one of this repository's images from ghcr.io and turn it into a SIF, on a compute node (Pawsey asks for pulls there, never on a
# login node).
#   sbatch --account=<project> pull_image.sbatch IMAGE [DEST_DIR]
#     IMAGE     e.g. ghcr.io/gbouras13/setonix-ports/af3jax:main, a release tag (af3jax:3.1.4-1) or a digest (@sha256:...) from the
#               Actions summary
#     DEST_DIR  default $MYSCRATCH/singularity. On /scratch the SIF is purged 21 days after it was last read; it is one file, so
#               rerunning this job brings it back. $MYSOFTWARE/singularity is not purged, if your quota has room (~5 GB). $MYSCRATCH
#               follows your DEFAULT project: if the image belongs elsewhere, give the directory (/scratch/<project>/$USER/singularity).
# Then: the image's provenance and its own self-check without a GPU (af3jax / colabfold / proteinmpnn check --no-gpu; mmseqs version).
# The GPU check runs in a GPU job (af3jax check, colabfold check, proteinmpnn check; mmseqs2-hip: ports/mmseqs2-hip/slurm/test_gpu.sbatch).
set -euo pipefail
IMAGE=${1:?usage: sbatch pull_image.sbatch IMAGE [DEST_DIR]}
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
singularity inspect --labels "$SIF" | grep -E "org.opencontainers.image.(title|source|revision|version|created)" || true
singularity exec "$SIF" sh -c 'cat /opt/*/source.txt /opt/*/kit.txt /opt/*/wheel.txt /opt/*/build.txt 2>/dev/null' || true

echo; echo "=== self-check without a GPU ==="
singularity exec "$SIF" sh -c 'for c in af3jax colabfold proteinmpnn; do command -v "$c" > /dev/null && exec "$c" check --no-gpu; done
                                command -v mmseqs > /dev/null && exec mmseqs version; echo "no known entry point"; exit 1'
echo; echo "SIF ready: $SIF   (run it without --rocm)"
