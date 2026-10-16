#!/bin/bash
# fetch-weights.sh <dir> -- the four files of the Boltz-2 cache, from Hugging Face (boltz-community; MIT licence, public, not
# gated), each checked against the kit's stock/PINS.json sha256 before it is moved into place. Run on a compute node (slurm/
# setup.sbatch). Idempotent: a file already present with the pinned digest is kept. A failed or mismatched download leaves
# only <file>.part behind, never a wrong file under the real name.
#
# The same bytes upstream's own downloader fetches (boltz/main.py download_boltz2: model-gateway.boltz.bio first, Hugging Face
# as its fallback); the Hugging Face LFS etags equal the kit's pins (checked 2026-10-03: boltz2_conf.ckpt 090e82ac..., 2.29 GB;
# boltz2_aff.ckpt dcc5cd37..., 2.06 GB; mols.tar 39e076d9..., 1.86 GB; ccd.pkl 2d3b2f03..., 0.35 GB).
#
# Not done here: unpacking mols.tar. Its member files would cost tens of thousands of /scratch inodes (the user's quota was at
# ~93-94% of 1,000,000); scripts/stage-cache.sh unpacks it per job on node-local NVMe instead.
set -uo pipefail
W=${1:?usage: fetch-weights.sh <dir>}
D=${B2S_ROOT:-/scratch/pawsey1018/gbouras/boltz2-setonix}
PINS=$D/kit/boltz2/stock/PINS.json
REV2=6fdef46d763fee7fbb83ca5501ccceff43b85607          # boltz-community/boltz-2 revision the digests were read from (2026-10-03)
PY=$D/venv/bin/python; [ -x "$PY" ] || PY=python3
mkdir -p "$W"
url_of() {
  case "$1" in
    ccd.pkl) echo "https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl" ;;   # upstream's CCD_URL (main.py:36)
    *)       echo "https://huggingface.co/boltz-community/boltz-2/resolve/$REV2/$1" ;;
  esac
}
want_of() { "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["weights"]["files"][sys.argv[2]]["sha256"])' "$PINS" "$1"; }
RC=0
for f in boltz2_conf.ckpt boltz2_aff.ckpt ccd.pkl mols.tar; do
  want=$(want_of "$f") || { echo "[fetch] no pinned sha256 for $f in $PINS" >&2; RC=1; continue; }
  if [ -f "$W/$f" ]; then
    have=$(sha256sum "$W/$f" | cut -d' ' -f1)
    if [ "$have" = "$want" ]; then echo "[fetch] $f present, sha256 OK"; continue; fi
    echo "[fetch] $f present but sha256 $have != pinned $want: refetching (old copy kept as $f.bad)"
    mv -f "$W/$f" "$W/$f.bad"
  fi
  t0=$SECONDS
  curl -L --fail --silent --show-error --retry 5 --retry-delay 10 -C - -o "$W/$f.part" "$(url_of "$f")" || { echo "[fetch] $f: curl failed" >&2; RC=1; continue; }
  have=$(sha256sum "$W/$f.part" | cut -d' ' -f1)
  if [ "$have" != "$want" ]; then echo "[fetch] $f: sha256 $have != pinned $want (kept as $f.part)" >&2; RC=1; continue; fi
  mv -f "$W/$f.part" "$W/$f"
  echo "[fetch] $f OK ($(stat -c %s "$W/$f") bytes, $((SECONDS - t0)) s)"
done
ls -la "$W"
exit $RC
