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
#   sbatch --account=<project> pull_image.sbatch IMAGE [DEST_DIR] [--gzip]
#     IMAGE     e.g. ghcr.io/gbouras13/setonix-ports/af3jax:main, a release tag (af3jax:3.1.4-1) or a digest (@sha256:...) from the
#               Actions summary. A tag is resolved to its digest first and pulled by that digest.
#     DEST_DIR  default $MYSCRATCH/singularity. On /scratch the SIF is purged 21 days after it was last read; it is one file, so
#               rerunning this job brings it back. $MYSOFTWARE/singularity is not purged, if your quota has room (~7 GB). $MYSCRATCH
#               follows your DEFAULT project: if the image belongs elsewhere, give the directory (/scratch/<project>/$USER/singularity).
#     --gzip    keep the SIF as Singularity writes it, instead of re-compressing it with lz4.
# lz4: Singularity's own mksquashfs writes gzip only, and Setonix's kernel decompresses a squashfs on one thread, so processes that start
# together queue for their libraries. With lz4, 16 ProteinMPNN workers started 11 s sooner and designed the same bytes (job 50681452;
# ports/proteinmpnn/README.md, Start-up). The conversion streams `unsquashfs -pf` into `mksquashfs -pf`, so every file, owner, mode and
# time is the image's own; the job proves it by hashing that stream before and after, checks that the new SIF starts and inspects like
# the pulled one, and keeps the gzip SIF if any step fails. About 2 minutes and 1.2 GB more SIF (job 50707295: 118 s, 5.2 -> 6.4 GB).
# Singularity prints an INFO line about lz4 whenever such a SIF starts; it is harmless, and `singularity -q` hides it.
# Beside the SIF, <name>.provenance.txt records what it is: the image and its digest, the source revision, and the conversion.
# Then: the image's own self-check without a GPU (af3jax / colabfold / proteinmpnn / boltz2 / protenix / opendde / esmfold2 check
# --no-gpu; mmseqs version). The GPU check runs in a GPU job (af3jax, colabfold, proteinmpnn, boltz2, protenix, opendde and esmfold2 check; mmseqs2-hip:
# ports/mmseqs2-hip/slurm/test_gpu.sbatch).
set -euo pipefail
GZIP=0; ARGS=()
for a in "$@"; do case "$a" in --gzip) GZIP=1 ;; *) ARGS+=("$a") ;; esac; done
IMAGE=${ARGS[0]:?usage: sbatch pull_image.sbatch IMAGE [DEST_DIR] [--gzip]}
DEST=${ARGS[1]:-${MYSCRATCH:?MYSCRATCH is not set}/singularity}
case "$IMAGE" in docker://*) IMAGE=${IMAGE#docker://};; esac
name=$(basename "${IMAGE%%@*}" | tr ':' '_')
case "$IMAGE" in *@sha256:*) d=${IMAGE##*@sha256:}; name="${name}_${d:0:12}";; esac
SIF=$DEST/$name.sif
NOTE=$DEST/$name.provenance.txt
mkdir -p "$DEST"
echo "=== $(hostname) job ${SLURM_JOB_ID:-?} $(date)  image docker://$IMAGE -> $SIF ==="

# Layers and the unpacked root filesystem (~25 GB, ~60k files) go to node-local /tmp, never to the 1 GB $HOME (Singularity's default
# cache) and never to /scratch (its inode quota). If /tmp is memory-backed on this node type, --mem above covers it (the pull's ~25 GB,
# then the conversion's ~17 GB).
T=/tmp/image-pull.${SLURM_JOB_ID:-$$}
mkdir -p "$T/cache" "$T/tmp"
trap 'rm -rf "$T"' EXIT
echo "node-local /tmp: $(df -h /tmp | awk 'NR==2 {print $4 " free of " $2 " (" $1 ")"}')"

module load singularity/4.1.0-nohost > /dev/null 2>&1 || { echo "ABORT: no singularity/4.1.0-nohost module"; exit 3; }
# after the module, which sets its own cache directory under $MYSOFTWARE
export SINGULARITY_CACHEDIR=$T/cache SINGULARITY_TMPDIR=$T/tmp APPTAINER_CACHEDIR=$T/cache APPTAINER_TMPDIR=$T/tmp
echo "singularity: $(singularity --version)"

# The digest this SIF is made from. A ghcr.io tag is resolved here (anonymous token) and pulled by digest, so the note records exactly
# what was pulled even if the tag moves later.
DIGEST="" SRC=$IMAGE
case "$IMAGE" in
  *@sha256:*) DIGEST=sha256:${IMAGE##*@sha256:} ;;
  ghcr.io/*)
    ref=${IMAGE#ghcr.io/}; repo=$ref; tag=latest
    case "${ref##*/}" in *:*) repo=${ref%:*}; tag=${ref##*:} ;; esac
    tok=$(curl -fsS "https://ghcr.io/token?scope=repository:$repo:pull" | sed -n 's/.*"token":"\([^"]*\)".*/\1/p') || tok=""
    DIGEST=$(curl -fsSI -H "Authorization: Bearer $tok" \
               -H "Accept: application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json" \
               "https://ghcr.io/v2/$repo/manifests/$tag" | tr -d '\r' | awk -F': ' 'tolower($1) == "docker-content-digest" {print $2}') || DIGEST=""
    case "$DIGEST" in sha256:*) SRC=ghcr.io/$repo@$DIGEST; echo "tag $tag = $DIGEST" ;; *) DIGEST=""; echo "could not resolve $tag; pulling the tag" ;; esac ;;
esac

t0=$SECONDS
PULLED=$T/pulled.sif
# ghcr.io drops an HTTP/2 stream now and then on a big layer (job 50659692: "stream error ... PROTOCOL_ERROR" on proteinmpnn's 3.7 GiB
# layer), so three tries. Layers already fetched stay in the cache above, so a retry fetches only what failed.
for try in 1 2 3; do
  singularity pull --force "$PULLED" "docker://$SRC" && break
  rm -f "$PULLED"
  [ "$try" -lt 3 ] || { echo "ABORT: the pull failed 3 times"; exit 1; }
  echo "pull attempt $try failed; retrying in 30 s"; sleep 30
done
echo "pulled in $((SECONDS - t0)) s ($(du -h "$PULLED" | cut -f1), gzip squashfs)"
rm -rf "$T/cache"   # the layers: only a retry needed them, and the conversion wants the room

# Re-compress the SIF's squashfs with lz4. Returns non-zero, with the reason in $why, if any step fails; the caller then keeps the
# pulled SIF. (Called in an `if`, so set -e is off in here: every step is checked.)
MKSQ=/usr/bin/mksquashfs UNSQ=/usr/bin/unsquashfs
insp () { { singularity inspect --labels "$1"; singularity inspect --environment "$1"; singularity inspect --runscript "$1"; } 2> /dev/null; }
lz4_convert () {  # $1 the pulled SIF, $2 the SIF to write
  local in=$1 out=$2 fsid="" id t o st hp help
  local objs=()
  help=$("$MKSQ" -help 2>&1 || true)
  case "$help" in *$'\tlz4'*) ;; *) why="$MKSQ has no lz4"; return 1 ;; esac
  [ -x "$UNSQ" ] || { why="no $UNSQ"; return 1; }
  # its objects: the squashfs, and the small ones carried over as they are (the definition file, Singularity's metadata, the OCI config)
  while IFS='|' read -r id _ _ _ t; do
    id=${id// /}
    case "$t" in
      *Squashfs*) fsid=$id ;;
      *Def.FILE*) objs+=("$id:1") ;;
      *JSON.Generic*) objs+=("$id:6") ;;
      *) why="an object this job does not know: $id ($t)"; return 1 ;;
    esac
  done < <(singularity sif list "$in" | grep -E '^[0-9]+ +\|')
  [ -n "$fsid" ] || { why="no squashfs partition"; return 1; }
  singularity sif dump "$fsid" "$in" > "$T/gzip.sqsh" || { why="sif dump"; return 1; }
  # the image as a pseudo file (paths, owners, modes, times, contents): hashed once as it is, and once as it goes into the lz4 squashfs
  ( "$UNSQ" -pf - "$T/gzip.sqsh" 2> /dev/null | sha256sum | cut -d' ' -f1 > "$T/before.sha" ) & hp=$!
  "$UNSQ" -pf - "$T/gzip.sqsh" 2> "$T/unsquashfs.log" | "$MKSQ" - "$T/lz4.sqsh" -pf - -comp lz4 -noappend -no-progress > "$T/mksquashfs.log" 2>&1
  st="${PIPESTATUS[*]}"; wait "$hp"
  rm -f "$T/gzip.sqsh"
  [ "$st" = "0 0" ] || { why="unsquashfs | mksquashfs exited $st: $(tail -c 300 "$T/mksquashfs.log" "$T/unsquashfs.log" | tr '\n' ' ')"; return 1; }
  BEFORE=$(cat "$T/before.sha"); AFTER=$("$UNSQ" -pf - "$T/lz4.sqsh" 2> /dev/null | sha256sum | cut -d' ' -f1)
  [ -n "$BEFORE" ] && [ "$BEFORE" = "$AFTER" ] || { why="the pseudo-file streams differ ($BEFORE before, $AFTER after)"; return 1; }
  # a new SIF: the small objects in their order, then the lz4 squashfs as the primary system partition; all in group 1, as pulled
  rm -f "$out"
  singularity sif new "$out" > /dev/null || { why="sif new"; return 1; }
  for o in "${objs[@]}"; do
    singularity sif dump "${o%%:*}" "$in" > "$T/object" \
      && singularity sif add --datatype "${o##*:}" --groupid 1 "$out" "$T/object" > /dev/null || { why="sif add, object ${o%%:*}"; return 1; }
  done
  singularity sif add --datatype 4 --parttype 2 --partfs 1 --partarch 2 --groupid 1 "$out" "$T/lz4.sqsh" > /dev/null || { why="sif add, squashfs"; return 1; }
  rm -f "$T/lz4.sqsh"
  singularity exec "$out" true > /dev/null 2>&1 || { why="the new SIF does not start"; return 1; }
  [ "$(insp "$in")" = "$(insp "$out")" ] || { why="singularity inspect (labels, environment, runscript) differs from the pulled SIF"; return 1; }
  singularity sif list "$out" | grep -E '^[0-9]+ +\|' | sed 's/^/  /'
}

t0=$SECONDS; why=""
if [ "$GZIP" = 1 ]; then
  KIND="gzip, as Singularity wrote it (--gzip)"; cp "$PULLED" "$SIF.partial"
elif lz4_convert "$PULLED" "$SIF.partial"; then
  KIND="lz4, re-compressed from the pulled gzip SIF by $("$MKSQ" -version 2>&1 | head -1) (unsquashfs -pf | mksquashfs -pf): every file, owner, mode and time is the image's own; sha256 of that pseudo-file stream, the same before and after: $BEFORE"
  echo "re-compressed with lz4 in $((SECONDS - t0)) s ($(du -h "$SIF.partial" | cut -f1)); pseudo-file streams identical"
else
  echo "lz4 conversion FAILED ($why): keeping the gzip SIF"
  KIND="gzip, as Singularity wrote it: the lz4 conversion failed ($why)"; rm -f "$SIF.partial"; cp "$PULLED" "$SIF.partial"
fi
mv -f "$SIF.partial" "$SIF"
REV=$(singularity inspect --labels "$SIF" 2> /dev/null | awk '$1 == "org.opencontainers.image.revision:" {print $2}')
{
  echo "sif        $SIF"
  echo "image      $IMAGE"
  echo "digest     ${DIGEST:-not resolved (pulled by tag)}"
  echo "revision   ${REV:-unknown} (org.opencontainers.image.revision)"
  echo "pulled     $(date -Is), job ${SLURM_JOB_ID:-?} on $(hostname), $(singularity --version)"
  echo "squashfs   $KIND"
  echo "sha256     $(sha256sum "$SIF" | cut -d' ' -f1) (this SIF)"
} > "$NOTE"

echo; echo "=== provenance ($NOTE) ==="
cat "$NOTE"
singularity inspect --labels "$SIF" | grep -E "org.opencontainers.image.(title|source|revision|version|created)" || true
singularity exec "$SIF" sh -c 'cat /opt/*/source.txt /opt/*/kit.txt /opt/*/wheel.txt /opt/*/build.txt 2>/dev/null' || true

echo; echo "=== self-check without a GPU ==="
singularity exec "$SIF" sh -c 'for c in af3jax colabfold proteinmpnn boltz2 protenix opendde esmfold2; do command -v "$c" > /dev/null && exec "$c" check --no-gpu; done
                                command -v mmseqs > /dev/null && exec mmseqs version; echo "no known entry point"; exit 1'
echo; echo "SIF ready: $SIF   (run it without --rocm)"
