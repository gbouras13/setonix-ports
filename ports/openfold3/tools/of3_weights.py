#!/usr/bin/env python3
"""of3_weights.py fetch|check DIR -- the OpenBind checkpoint OpenFold3 0.5.0 runs by default, as the native port has it.

    fetch DIR   download of3-ob-2025-06-30-174k.pt into DIR with OpenFold3's own downloader (openfold3.entry_points.parameters,
                unsigned S3 from its public bucket s3://openfold3-data/openfold3-parameters/), first into DIR/.of3-fetch.part, and move
                it into place only once it is checked. A file already in DIR is checked and kept, never fetched again
    check DIR   check DIR/of3-ob-2025-06-30-174k.pt; reads nothing from the network

The check is against the S3 object as published (HEAD of 2026-10-21; Last-Modified 2026-07-24), three ways in one read of the file:
  - its size, 2,287,872,989 bytes, the native port's copy too;
  - its full-object CRC64NVME (x-amz-checksum-crc64nvme), the value OpenFold3's own s3_file_matches_local() compares, computed with
    the same awscrt function;
  - its ETag, a 273-part multipart upload: the MD5 of the 8 MiB parts' MD5s (reported, not decisive: it rests on the uploader's part
    size, which S3 does not publish).
and prints the file's sha256 for the record. Exit 0 when the size and the CRC64NVME hold, 1 otherwise.
"""
import hashlib
import os
import shutil
import sys
from pathlib import Path

NAME = "openbind-2025-06-30-174k"
FILE = "of3-ob-2025-06-30-174k.pt"
SIZE = 2287872989
CRC64NVME = "eXdSi7ctLsk="
ETAG = "e3def01bcb8671e21583166512834340-273"
PART = 8 * 1024 * 1024


def digests(path):
    import base64
    from awscrt import checksums
    crc, sha, parts, n = 0, hashlib.sha256(), [], 0
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(PART)
            if not chunk:
                break
            n += len(chunk)
            crc = checksums.crc64nvme(chunk, crc)
            sha.update(chunk)
            parts.append(hashlib.md5(chunk).digest())
    etag = "%s-%d" % (hashlib.md5(b"".join(parts)).hexdigest(), len(parts))
    return n, base64.b64encode(crc.to_bytes(8, byteorder="big")).decode(), etag, sha.hexdigest()


def check(path):
    if not path.is_file():
        print("[weights] %s: absent" % path)
        return False
    size, crc, etag, sha = digests(path)
    ok = size == SIZE and crc == CRC64NVME
    print("[weights] %s: size %d (%s), crc64nvme %s (%s), etag %s (%s)"
          % (path, size, "ok" if size == SIZE else "want %d" % SIZE, crc, "ok" if crc == CRC64NVME else "want " + CRC64NVME,
             etag, "ok" if etag == ETAG else "want %s; reported, not decisive" % ETAG))
    print("[weights] sha256 %s" % sha)
    print("[weights] %s" % ("CHECKPOINT OK: the S3 object as published" if ok else "CHECKPOINT DIFFERS from the S3 object"))
    return ok


def fetch(d):
    target = d / FILE
    if target.exists():
        print("[weights] %s is present: checking it, not fetching" % target)
        return check(target)
    part = d / ".of3-fetch.part"
    if part.exists():
        shutil.rmtree(part)
    from openfold3.entry_points.parameters import download_model_parameters
    print("[weights] fetching %s with OpenFold3's own downloader into %s" % (NAME, part), flush=True)
    download_model_parameters(part, NAME, force_download=True, skip_confirmation=True)
    if not check(part / FILE):
        print("[weights] REFUSED: the download is not the published object; left in %s" % part)
        return False
    os.replace(part / FILE, target)
    part.rmdir()
    print("[weights] %s in place" % target)
    return True


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("fetch", "check"):
        sys.exit(__doc__)
    d = Path(sys.argv[2])
    ok = fetch(d) if sys.argv[1] == "fetch" else check(d / FILE)
    sys.exit(0 if ok else 1)
