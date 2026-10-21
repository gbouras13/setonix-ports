#!/bin/bash
# ef2-stage.sh -- copy the pinned weights (27.5 GB, the HF hub layout under $D/weights) onto the node's local NVMe (/tmp) once per job,
# and point HF_HOME at the copy. SOURCE it after scripts/ef2-env.sh, in the batch script, before launching any process:
#
#   source scripts/ef2-env.sh && source scripts/ef2-stage.sh      (the job's EXIT trap should remove $EFS_WEIGHTS_LOCAL)
#
# Why: upstream loads the checkpoints through safetensors' mmap, and page faults off Lustre ran at a few MB/s per process (smoke2
# 50986746: four processes still in filemap_fault / cl_sync_io_wait after 23 min, ~8 GB resident each; smoke1's single process took
# ~520 s to load). Sequential reads of the same files run at ~330 MB/s, and a node-local copy serves the faults from NVMe / page cache.
# The files are copied unchanged (sizes checked; mtimes kept); the kit digests the copy against stock/PINS.json as it does any weights
# root (its digest memo is keyed by path and inode, so the first process of a job digests the copy afresh).
D=${EFS_ROOT:-/scratch/pawsey1018/gbouras/esmfold2-setonix}
SRC=${EFS_WEIGHTS_SRC:-$D/weights}
DST=${EFS_WEIGHTS_LOCAL:-/tmp/efs-weights-$USER-${SLURM_JOB_ID:-nojob}}
export EFS_WEIGHTS_LOCAL=$DST
# Processes started side by side stage once: the first holds the lock and copies, the others wait and reuse its copy.
_lk=0; if command -v flock > /dev/null 2>&1 && exec 9> "$DST.lock"; then flock 9 && _lk=1; fi
if [ -f "$DST/.staged" ]; then
  echo "[efs-stage] reusing $DST ($(du -sh "$DST" 2>/dev/null | cut -f1))"
else
  _t0=$SECONDS
  mkdir -p "$DST" && chmod 700 "$DST"
  (cd "$SRC" && find hub -type d) | while read -r d; do mkdir -p "$DST/$d"; done
  (cd "$SRC" && find hub -type f -print0) | xargs -0 -P "${EFS_STAGE_JOBS:-6}" -I{} cp --preserve=timestamps "$SRC/{}" "$DST/{}"
  _bad=0
  while read -r f; do
    [ "$(stat -c %s "$SRC/$f")" = "$(stat -c %s "$DST/$f" 2>/dev/null)" ] || { echo "[efs-stage] SIZE MISMATCH $f" >&2; _bad=1; }
  done < <(cd "$SRC" && find hub -type f)
  if [ "$_bad" = 0 ]; then
    touch "$DST/.staged"
    _gb=$(du -sb "$DST" | cut -f1)
    echo "[efs-stage] staged $SRC -> $DST: $((_gb / 1000000)) MB in $((SECONDS - _t0)) s"
  else
    echo "[efs-stage] staging FAILED; HF_HOME stays $SRC" >&2
    DST=$SRC
  fi
fi
export HF_HOME=$DST
[ "$_lk" = 1 ] && { flock -u 9; exec 9>&-; }
unset _t0 _bad _gb _lk
