#!/usr/bin/env python3
"""Peers in flight for the pair block's two in-place shard transposes (env-gated; off by default).

WHY
  pair_block_ (opt_core.mem.rowpair.pairstack) transposes the fp32 pair shard before the ending triangle attention and back after
  it. Under the launcher's ROWPAIR_TRANSPOSE_INPLACE=1 that is ring.transpose_shard_inplace_, which swaps blocks pairwise on a
  symmetric schedule (round d: peer = rank XOR d). pair_block_ passes no peers_in_flight, so the 7 rounds at P=8 run one after
  another. The trunk profile at 6,930 res / P=8 (job 50387728) puts the two transposes at 0.243 s of a 3.12 s pairformer block
  (7.8%), nearly all of it RCCL. Ring benchmark round 5 (job 50388002) timed the kit's own function at those shapes:
  124 ms per transpose with 1 peer in flight, 126 with 2, 107 with 4, 65 ms with all 7 (1.90x; repeated at the end of the job).
  Every k was bitwise identical to k=1, one transpose and a round trip.

WHAT
  ROWPAIR_TRANSPOSE_PEERS=auto: every round in flight at once (next_pow2(P) - 1 send + receive pairs of n_loc x Rmax x C staging,
  5.4 GiB at 6,930 / P=8) when that staging is at most ROWPAIR_TRANSPOSE_PEERS_FRAC (default 0.5) of the free device memory
  agreed over ranks (dist.agreed_free_bytes: the minimum over ranks, so every rank runs the same rounds), else one peer per round
  as before. Decided once per block, before the first transpose; both transposes use it.
  ROWPAIR_TRANSPOSE_PEERS=<k>: k pairs, no memory check. Unset, 1, 0 or off: one peer per round, exactly as before.
  The transposes copy values; which rounds share a batched exchange changes only when the bytes move, so outputs are bitwise
  those of k=1. Only the in-place form is touched (the out-of-place transpose_shard path is unchanged).
  Each rank logs its decision once per (N, k): "[transpose_peers] N=6930: 7 peer(s) in flight per round group (auto: staging ...)".
  Validated 2026-10-05 (job 50409167): the pinned hexamer bitwise; the 18-mer trunk 697.7 -> 674.6 s (-3.3%).
  A default of scripts/predict-sharded.sh since 2026-10-05 (the user's choice).

USAGE  python3 patch_transpose_peers.py --apply | --check | --revert [path]
"""
import os, sys, pathlib

P = pathlib.Path('/scratch/pawsey1018/gbouras/af3-setonix/opt_core_src/opt_core/opt_core/mem/rowpair/pairstack.py')
if len(sys.argv) > 2:
    P = pathlib.Path(sys.argv[2])
MARK = '# [transpose_peers]'
BACKUP = P.with_suffix('.py.pre_transpose_peers')

ANCHOR = "def pair_block_(fns: PairBlockFns, z_shard, mask_shard, layout: Layout, *, s=None, maskT_shard=None, transition_mask: bool = True,\n"
HELPER = '''# [transpose_peers] ROWPAIR_TRANSPOSE_PEERS: peers in flight for pair_block_'s two in-place shard transposes (ring.transpose_shard_inplace_,
# the ROWPAIR_TRANSPOSE_INPLACE=1 path). Unset / 1 / 0 / off: one peer per round, as before. auto: every round at once when its staging
# (2 x (next_pow2(P) - 1) blocks of n_loc x Rmax x C) is at most ROWPAIR_TRANSPOSE_PEERS_FRAC (default 0.5) of the free device memory
# agreed over ranks (dist.agreed_free_bytes, so all ranks run the same rounds), else one. <k>: k pairs. The transposes copy values, so
# outputs are bitwise those of one peer per round (job 50388002: 124 -> 65 ms per transpose at 6,930 res / P=8 with all 7 in flight).
_TPEERS_SAID = set()                                                     # (N, k) decisions already logged by this process


def _transpose_peers_say(N: int, k: int, text: str) -> None:
    if (N, k) in _TPEERS_SAID:
        return
    _TPEERS_SAID.add((N, k))
    try:
        from .dist import comm
        comm().log(f"[transpose_peers] N={N}: {k} peer(s) in flight per round group ({text})")
    except Exception:                                                     # noqa: BLE001 -- a log line never breaks a block
        print(f"[transpose_peers] N={N}: {k} peer(s) in flight per round group ({text})", flush=True)


def _transpose_peers(z, layout: Layout) -> int:
    word = os.environ.get("ROWPAIR_TRANSPOSE_PEERS", "").strip().lower()
    if word in ("", "1", "0", "off"):
        return 1
    P2 = 1
    while P2 < int(layout.P):
        P2 *= 2
    rounds = max(1, P2 - 1)
    if word != "auto":
        try:
            k = int(word)
        except ValueError:
            raise RowpairRefused(f"ROWPAIR_TRANSPOSE_PEERS={word!r}: expected auto or a positive integer")
        k = max(1, min(k, rounds))
        _transpose_peers_say(int(layout.N), k, f"ROWPAIR_TRANSPOSE_PEERS={word}")
        return k
    from .dist import agreed_free_bytes
    lead = 1
    for d in z.shape[:-3]:
        lead *= int(d)
    need = 2 * rounds * lead * int(layout.n_loc) * int(layout.Rmax) * int(z.shape[-1]) * z.element_size()
    free = agreed_free_bytes(z.device)
    frac = float(os.environ.get("ROWPAIR_TRANSPOSE_PEERS_FRAC", "").strip() or 0.5)
    k = rounds if (free is not None and need <= frac * free) else 1
    _transpose_peers_say(int(layout.N), k, f"auto: staging {need / 2 ** 30:.2f} GiB vs {frac:g} x agreed free "
                                           f"{'?' if free is None else f'{free / 2 ** 30:.2f}'} GiB")
    return k


'''
OLD_T1 = ("        from .ring import transpose_shard_inplace_                        # noqa: WPS433\n"
          "        zT = transpose_shard_inplace_(z, layout)\n")
NEW_T1 = ("        from .ring import transpose_shard_inplace_                        # noqa: WPS433\n"
          "        tpeers = _transpose_peers(z, layout)                              # [transpose_peers] decided once per block\n"
          "        zT = transpose_shard_inplace_(z, layout, peers_in_flight=tpeers)\n")
OLD_T2 = ("    if tin:\n"
          "        z = transpose_shard_inplace_(zT, layout)\n")
NEW_T2 = ("    if tin:\n"
          "        z = transpose_shard_inplace_(zT, layout, peers_in_flight=tpeers)   # [transpose_peers]\n")


def _apply(s):
    counts = {"anchor": s.count(ANCHOR), "t1": s.count(OLD_T1), "t2": s.count(OLD_T2)}
    if any(n != 1 for n in counts.values()):
        raise SystemExit(f'ANCHORS NOT FOUND EXACTLY ONCE ({counts}) -- not patching')
    for need in ('\nimport os\n', 'RowpairRefused'):
        if need not in s:
            raise SystemExit(f'pairstack.py lacks {need.strip()!r} -- not patching')
    new = s.replace(ANCHOR, HELPER + ANCHOR, 1).replace(OLD_T1, NEW_T1, 1).replace(OLD_T2, NEW_T2, 1)
    compile(new, str(P), 'exec')
    return new


def _write(text):
    tmp = P.with_suffix('.py.transpose_peers_tmp')
    tmp.write_text(text)
    os.replace(tmp, P)                                     # atomic: importers never see a half-written file


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else '--check'
    s = P.read_text()
    applied = MARK in s
    if mode == '--check':
        print(f'transpose_peers: applied={applied}  backup={BACKUP.exists()}')
        return 0
    if mode == '--apply':
        if applied:
            print('already applied'); return 0
        BACKUP.write_text(s)
        _write(_apply(s))
        print(f'applied (backup: {BACKUP.name})'); return 0
    if mode == '--revert':
        if not BACKUP.exists():
            print('no backup', file=sys.stderr); return 1
        _write(BACKUP.read_text())
        print('reverted'); return 0
    print('usage: --apply | --check | --revert [path]', file=sys.stderr); return 2


sys.exit(main())
