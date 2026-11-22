#!/bin/bash
# Fetch checkpoint/protenix-v2.pt into the weights root the way the kit's own
# prefetch does (protenix_opt/weights.py prefetch_checkpoint): transfer to
# <checkpoint>.part, compare its sha256 with stock/PINS.json, and only then
# rename it into place. Nothing loads the file before the digest matches.
#
# Source: upstream's URL answers HTTP 403 (kit README 'Setup'). The Hugging Face
# repo TMF001/protenix-v2-weights, approved by the user and used on 2026-09-23,
# answers HTTP 401 (first seen 2026-10-20). fredzzp/protenix-v2 carries
# checkpoint/protenix-v2.pt with an LFS sha256 and size equal to the pin
# (checked 2026-10-23); the user approved it on 2026-10-23.
# PXS_CHECKPOINT_URL=<url> asks any other source instead. The pin check below is
# what makes the source irrelevant.
set -euo pipefail
W=${1:?usage: fetch_checkpoint.sh <weights root>}
PIN_SHA=8f931f9774a396b67033d0e58628e1834f4a1448165e04254b40a780b0c0d599
PIN_BYTES=1859785497
URL=${PXS_CHECKPOINT_URL:-https://huggingface.co/fredzzp/protenix-v2/resolve/main/checkpoint/protenix-v2.pt}
F=$W/checkpoint/protenix-v2.pt
mkdir -p "$W/checkpoint"
if [ -f "$F" ]; then
  echo "[fetch] $F present; the kit's install --weights check judges it"
  exit 0
fi
echo "[fetch] $URL -> $F.part"
t0=$SECONDS
curl -fsSL --retry 5 --retry-delay 10 -C - -o "$F.part" "$URL"
sz=$(stat -c %s "$F.part")
echo "[fetch] transferred $sz bytes in $((SECONDS-t0)) s"
[ "$sz" = "$PIN_BYTES" ] || { echo "[fetch] REFUSED: size $sz is not the pin $PIN_BYTES; left at $F.part" >&2; exit 1; }
got=$(sha256sum "$F.part" | cut -d' ' -f1)
[ "$got" = "$PIN_SHA" ] || { echo "[fetch] REFUSED: sha256 $got is not the pin $PIN_SHA; left at $F.part, never loaded" >&2; exit 1; }
mv "$F.part" "$F"
echo "[fetch] sha256 $got = the pin; in place"
