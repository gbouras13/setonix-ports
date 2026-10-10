#!/usr/bin/env python
"""pms_launch.py -- `python -m proteinmpnn_opt <design|check|warm> ...` with the kit worker's command line routed through
scripts/pms_worker_boot.py (named gfx90a / ROCm patches, PMS_WORKER_PATCHES; see that file).

    python pms_launch.py design --mode exact --input DIR --out OUT [stock options] ...      (scripts/pms-run.sh is the shell form)

The kit's kit_run.run() starts the worker with `_launch` / `_launch_overlapped` as [python, <stage>/.../kit/mpnn_worker2[_lowmem].py, args];
this replaces those two functions of the loaded kit_run module with wrappers that turn such a command into
[python, pms_worker_boot.py, <that path>, args] and call the originals. Every other step (the core gate, activation, staging, the parse
step, the manifest, the EXIT line and exit code) is the kit's own code, unchanged; no kit file is edited. The CMD line the kit prints
names the worker as the kit launched it; the boot's first line says what it applied.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BOOT = os.path.join(HERE, "pms_worker_boot.py")

from proteinmpnn_opt import core_gate  # noqa: E402

core_gate()                                                                    # statement one of every kit entry (the core pin gate)

from proteinmpnn_opt import cli, kit_run  # noqa: E402


def _route(cmd):
    if len(cmd) >= 2 and os.path.basename(str(cmd[1])).startswith("mpnn_worker2") and str(cmd[1]).endswith(".py"):
        return [cmd[0], BOOT] + list(cmd[1:])
    return cmd


_orig_launch, _orig_overlapped = kit_run._launch, kit_run._launch_overlapped


def _launch(cmd, env, cwd, echo=True):
    return _orig_launch(_route(cmd), env, cwd, echo)


def _launch_overlapped(parse_cmd, worker_cmd, ready, env, echo=True):
    return _orig_overlapped(parse_cmd, _route(worker_cmd), ready, env, echo)


kit_run._launch = _launch
kit_run._launch_overlapped = _launch_overlapped

if __name__ == "__main__":
    sys.exit(cli.main(sys.argv[1:]))
