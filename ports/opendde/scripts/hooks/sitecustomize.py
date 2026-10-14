"""Per-rank JIT cache directories for OpenDDE's sharded line.

opt_core's launcher gives every rank process the SAME environment plus
ROWPAIR_RANK / ROWPAIR_WORLD (launch.rank_env), so all P ranks would share one
TRITON_CACHE_DIR and one MIOpen database. Concurrent writes to a single Triton
cache from 8 ranks is a known corruption source on this machine.

This runs at interpreter start (PYTHONPATH), touches nothing outside a rank
process, and is the only code this port injects into the kit's own launcher.
"""
import os

_rank = os.environ.get("ROWPAIR_RANK")
if _rank is not None and _rank.strip().isdigit():
    for _v in ("TRITON_CACHE_DIR", "MIOPEN_USER_DB_PATH", "MIOPEN_CUSTOM_CACHE_DIR",
               "TORCHINDUCTOR_CACHE_DIR"):
        _base = os.environ.get(_v)
        if _base:
            _p = os.path.join(_base, "rank%s" % _rank.strip())
            try:
                os.makedirs(_p, exist_ok=True)
            except OSError:
                continue
            os.environ[_v] = _p
