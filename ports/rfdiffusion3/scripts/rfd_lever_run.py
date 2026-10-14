"""Run upstream's own `rfd3 design` in the KIT interpreter with the levers named on the command line — one at a time.

    RFD_DET=1 python scripts/rfd_lever_run.py --levers RFD3_FZT -- design inputs=<spec> out_dir=<dir> seed=101 ...

Why this exists: a kit MODE is selected whole (`--mode exact` exports all four of its switches, and a switch preset at any
other value is refused by name), so a mode cannot tell us WHICH lever moved a design on this card. The levers themselves are
plain environment reads at the import of the overlay's modules (`hoist.py:18`, `layers/layer_utils.py` `[RFD3_FZT]` /
`[RFD3_INIT_CHUNK]`, `inference_sampler.py:610` for `RFD3_CUDAGRAPH`), so upstream's unchanged command line in the patched
interpreter, with one switch set and `RFDIFFUSION3_OPT` unset, engages exactly that lever and nothing else — no kit
activation, no census, no gate. Nothing in the kit tree is edited.

`RFD_DET=1` applies the kit's OWN deterministic recipe (`rfdiffusion3_opt.det.apply(1)`: the cuBLAS workspace, torch's
deterministic algorithms, the fixed-order scatter-mean), so a bisection run is comparable with `run.sh design --det 1`.
The two variables that must precede `import torch` are set by the wrapper script that calls this one.

At exit it prints each lever's own tally — `FZT_STATS`, `INIT_CHUNK_STATS`, `hoist.STATS`, the graph sampler's `GRAPH_STATS` —
so a run says how often the lever it was given actually served a call.
"""
from __future__ import annotations

import json
import os
import sys


def main() -> int:
    argv = sys.argv[1:]
    levers = []
    if argv and argv[0] == "--levers":
        levers = [v for v in argv[1].split(",") if v]
        argv = argv[2:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    for name in levers:                       # set before rfd3.model is imported, which is what the overlay reads
        os.environ[name] = "1"
    if os.environ.get("RFDIFFUSION3_OPT"):
        print("rfd_lever_run: RFDIFFUSION3_OPT is set — unset it: this runner engages levers, not a mode", file=sys.stderr)
        return 2
    det = (os.environ.get("RFD_DET") or "0").strip()
    if det == "1":
        from rfdiffusion3_opt import det as _det
        applied = _det.apply(1)
        print(f"[rfd-lever] det recipe: {applied}", file=sys.stderr)
    if (os.environ.get("RFD_DET_STRICT") or "0").strip() == "1":
        # The kit's recipe uses warn_only=True, so an op with no deterministic implementation still runs, just warned.
        # Strict mode makes torch RAISE and name it - the one diagnostic that says WHAT is not reproducible here.
        import torch
        torch.use_deterministic_algorithms(True, warn_only=False)
        print("[rfd-lever] STRICT determinism: torch.use_deterministic_algorithms(True, warn_only=False)", file=sys.stderr)
    print(f"[rfd-lever] levers={','.join(levers) or 'none'} det={det} argv={' '.join(argv)}", file=sys.stderr)

    from rfd3.cli import app
    sys.argv = ["rfd3"] + argv
    rc = 0
    try:
        app()
    except SystemExit as e:
        rc = int(e.code or 0)
    finally:
        stats = {}
        lu = sys.modules.get("rfd3.model.layers.layer_utils")
        for mod, attr in ((lu, "FZT_STATS"), (lu, "INIT_CHUNK_STATS"),
                          (sys.modules.get("rfd3.model.hoist"), "STATS"),
                          (sys.modules.get("rfd3.model.cudagraph_sampler"), "GRAPH_STATS")):
            v = getattr(mod, attr, None) if mod is not None else None
            if v is not None:
                stats[attr] = v
        print("[rfd-lever] STATS " + json.dumps(stats, default=str)[:4000], file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
