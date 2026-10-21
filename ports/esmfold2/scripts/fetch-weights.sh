#!/bin/bash
# fetch-weights.sh -- the three pinned weight snapshots (ESMC-6B, ESMFold2, ESMFold2-Fast: 16 files, 27.5 GB) into DIR, in Hugging
# Face's cache layout, then the kit's OWN check of every file. Needs the network (Setonix compute nodes have outbound HTTPS).
#
#   bash scripts/fetch-weights.sh DIR          (source scripts/ef2-env.sh first; the image's `esmfold2 fetch-weights DIR` does both)
#
# Files land under DIR/hub/models--<org>--<name>/snapshots/<commit>/, the commits and sizes read from the kit's stock/PINS.json. The
# transfer is curl, resumable (-C -), at the pinned snapshot commit of each repository; a file already there at its pinned size is kept.
# The JUDGE is the kit: `python -m esmfold2_opt.weights DIR` points refs/main at the pinned commits and digests every file against
# stock/PINS.json (exit 1 naming any file off its pin). scripts/setup.sh step 5 is the same procedure for the native tree; the RFdiffusion3
# port found upstream's own downloader (huggingface_hub) stalling on a compute node while curl answered at once.
set -euo pipefail
DIR=${1:?usage: fetch-weights.sh DIR}
[ -n "${MODEL_OPT:-}" ] && [ -f "$MODEL_OPT/stock/PINS.json" ] || { echo "[fetch] MODEL_OPT is not the kit's esmfold2 directory (source scripts/ef2-env.sh)" >&2; exit 2; }
mkdir -p "$DIR"
LIST=$(mktemp)
trap 'rm -f "$LIST"' EXIT
python - "$MODEL_OPT/stock/PINS.json" > "$LIST" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
for repo, w in sorted(p["weights"].items()):
    for name, d in sorted(w["files"].items()):
        print(repo, w["snapshot_commit"], name, d["size_bytes"], sep="\t")
PY
echo "[fetch] $(wc -l < "$LIST") files of $(cut -f1 "$LIST" | sort -u | tr '\n' ' ')into $DIR"
t0=$SECONDS; nfail=0
while IFS=$'\t' read -r repo commit name size; do
  dest=$DIR/hub/models--${repo/\//--}/snapshots/$commit/$name
  if [ -f "$dest" ] && [ "$(stat -c %s "$dest")" = "$size" ]; then echo "[fetch] present  $repo $name"; continue; fi
  mkdir -p "$(dirname "$dest")"
  t1=$SECONDS
  if curl -fsSL --retry 5 --retry-delay 10 --retry-connrefused -C - --speed-limit 1048576 --speed-time 120 \
          -o "$dest.part" "https://huggingface.co/$repo/resolve/$commit/$name" && mv -f "$dest.part" "$dest"; then
    got=$(stat -c %s "$dest")
    echo "[fetch] fetched  $repo $name $got B in $((SECONDS - t1)) s"
    [ "$got" = "$size" ] || { echo "[fetch] SIZE     $repo $name: $got != $size"; nfail=$((nfail + 1)); }
  else
    echo "[fetch] FAILED   $repo $name"; nfail=$((nfail + 1))
  fi
done < "$LIST"
echo "[fetch] curl: $((SECONDS - t0)) s, $nfail failed"
[ "$nfail" = 0 ] || exit 1
echo "[fetch] the kit's own check (refs/main -> the pinned commits, sha256 of every file against stock/PINS.json):"
python -m esmfold2_opt.weights "$DIR"
echo "[fetch] OK $(du -sh "$DIR" | cut -f1) in $DIR"
