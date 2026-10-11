#!/bin/bash
# stage-cache.sh -- this job's BOLTZ_CACHE on node-local NVMe: symlinks to the four pinned files in $B2S_WEIGHTS (default
# $D/weights), and mols/ unpacked from mols.tar. Prints the directory as its LAST line:
#
#   export BOLTZ_CACHE=$(bash scripts/stage-cache.sh | tail -1)
#
# Why: the kit (stack.py CACHE_FILES) and upstream's downloader (main.py download_boltz2) both require mols/ unpacked beside
# mols.tar, and upstream DOWNLOADS whatever is missing. mols/ holds one pickle per CCD component, tens of thousands of files:
# on /scratch that is a large slice of the user's inode quota (93-94% of 1,000,000 used on 2026-10-03). On the node's NVMe it
# costs nothing and is gone with the job. The kit's cache check follows symlinks (os.path.exists / isfile), and its digest memo
# keys entries by the file's realpath, so the memo in $MODEL_OPT_WEIGHTS_DIGEST_DIR keeps serving across jobs.
set -euo pipefail
D=${B2S_ROOT:-/scratch/pawsey1018/gbouras/boltz2-setonix}
W=${B2S_WEIGHTS:-$D/weights}          # the image keeps no weights: its entry point sets B2S_WEIGHTS to the directory you give it
C=${B2S_LOCAL:-/tmp/b2s-$USER-${SLURM_JOB_ID:-nojob}}/boltz_cache
mkdir -p "$C"
for f in boltz2_conf.ckpt boltz2_aff.ckpt ccd.pkl mols.tar; do
  [ -f "$W/$f" ] || { echo "[stage-cache] missing $W/$f (run slurm/setup.sbatch, or the image's fetch-weights)" >&2; exit 1; }
  ln -sfn "$W/$f" "$C/$f"
done
if [ ! -d "$C/mols" ]; then
  t0=$SECONDS
  tar -C "$C" -xf "$W/mols.tar"
  echo "[stage-cache] mols/ unpacked in $((SECONDS - t0)) s: $(find "$C/mols" -type f | wc -l) files, $(du -sh "$C/mols" | cut -f1)" >&2
fi
[ -d "$C/mols" ] || { echo "[stage-cache] mols.tar did not unpack to mols/" >&2; exit 1; }
echo "$C"
