#!/usr/bin/env python3
"""b2s_launch.py -- the Boltz-2 kit's own entry (``python -m boltz2_opt <verb> ...``) with opt-in Setonix (gfx90a / ROCm) patches
applied to the kit's MODE TABLE in this process only.

    python b2s_launch.py pred --mode big --input X.yaml --out_dir OUT [...]    (the same arguments as ``python -m boltz2_opt``)

Why here and nowhere else: the parent process (``python -m boltz2_opt pred``) is where every resolution of a mode's row happens
(boltz2_opt.worker.run -> modes.resolve, stack.gate -> modes.set_card, stack.worker_command / kernels_opts -> modes.worker_args /
modes.kernels_expected). The worker it launches receives the resolved row through its command line (--kernels, --kernels-expect,
--attach) and its environment (the row's switch words), so a patch of the table here reaches the worker without touching it.
Nothing on disk changes: the kit checks its own files' sha256 at every launch (stack.check_kit_files).

Switches (the B2S_ prefix is outside every name family the kit strips or refuses):

  B2S_KERNELS=off   run exact / fast / big with the worker's ``--kernels off``, the kit-side equivalent of stock's ``--no_kernels``.
                    Why: the three rows carry ``"kernels": "on"`` (modes.py MODES), so the KERNELS census expects upstream's two
                    cuEquivariance accelerators ``engaged`` (modes.kernels_expected; at n_gpu > 1 only the TriMul is replaced), and
                    cuEquivariance does not exist on ROCm: every kit mode would end ``KERNELS-REFUSED ... cueq_triatt=absent; exit 5``
                    at its first predict_step (kernels.py REQUIRE). With this the census expects
                    ``off-by-route:worker--kernels=off`` for both, exactly as for ``pred --mode off --no_kernels``.

  B2S_CARD_DROPS="<mode>:<lever>:<reason>;..."   levers that leave a row ON THIS CARD, BY NAME, through the kit's own table
                    (modes.CARD_DROPS under the probed compute capability, the one nvidia-smi reports: B2S_SHIM_CC, default 9.0).
                    They print as ``card_off=<lever>:<reason>`` and one ``state=off`` LEVER line each, like the kit's own sm_80 /
                    sm_100 drops. modes.take_off removes each lever's switch words and attachment, and every lever that rides it.

  B2S_WORKER_PATCHES="transition_no_v2,trimul_no_native"   OPT-IN worker-side patches (scripts/b2s_worker_boot.py): the kit's worker
                    command line (stack.worker_command: the one-GPU worker and every rank) is rewritten to start through the boot
                    script, which installs the named patches and then runs `-m boltz2_opt.worker_launch` with the kit's own arguments.
                    The patches refuse opt_core rows by name at selection time (see that file); default: none, the command unchanged.

  B2S_BIG_BASE=flash   compose `big`'s row on the kit's OTHER measured pair-stack base (modes.BIG_ATTN_BASES = flash | pairtrack;
                    shipped: pairtrack, chosen by measurement on H100 with cuEquivariance). The row is the kit's own composition:
                    modes.py's source executed once more in a private namespace with that one constant changed, and big's
                    `levers` / `env` / `attach` taken from it (flash: - pairblock, pairblock_c64, fused_transition, pairfuse,
                    condproj and their words; + BOLTZ_TRIATTN=flash, BOLTZ_TRIATTN_MIN_TOKENS=300). Why: on ROCm the pair track's
                    attention core is cuEquivariance's and steps aside under --kernels off, so triangle attention runs stock's
                    chunked code on one GCD (B6/B7: flash_triattn calls=0); the flash base serves it with opt_core's flash kernel,
                    the one that serves the sharded line on gfx90a.

  B2S_TP_SPEED=0    opt out of the sharded line's speed words, which are ON BY DEFAULT for `pred ... --n_gpu P` with P > 1 (the user's
                    decision, 2026-10-08): NCCL_NCHANNELS_PER_PEER=8 and TORCH_NCCL_HIGH_PRIORITY=1 (set only where the caller has
                    not set them) and the boot word triattn_tiles added to B2S_WORKER_PATCHES (PTX_TRIATTN_CONFIG, default
                    128,32,2,4,1,1, in every rank). Measured byte-identical to the line without them, same peak memory: 1.79x at P=8 /
                    5,390 tokens (speed8 50463489), 1.11x at P=2 / 1,155 (valid 4 50481534).

  B2S_BIG_FLASH=0   opt out of big's one-GCD flash recipe, which is ON BY DEFAULT for `pred --mode big` on one GCD (the user's decision,
                    2026-10-09): B2S_BIG_BASE=flash, OF3T_FLASH_IP=ieee and B2S_TRIATTN_CONFIG=128,32,1,4,1,1 (set only where the caller
                    has not set them) and the boot words trimul_v4_only + triattn_tiles added to B2S_WORKER_PATCHES. An explicit
                    B2S_BIG_BASE=pairtrack keeps the shipped base and adds nothing. Measured (speed1b 50480198, valid 4-6, precision
                    50538802): 2.6x stock on the HK97 trimer, 6.1-6.7x on the 6-mer, less memory, the same one-GCD ceiling; within
                    0.06-0.15 A of the shipped big on the trimer; its triangle attention is more precise than stock's path (0.83-0.90x
                    the error vs float64) but rounds differently, so on low-confidence inputs a seed can land elsewhere (6-mer).
                    Card drops naming the pair track's own levers (pairblock, pairblock_c64, fused_transition, pairfuse, condproj),
                    which the flash row does not carry, are skipped with a note.

  B2S_LAUNCH_QUIET=1   no ``[b2s-launch]`` line.
"""
from __future__ import annotations

import os
import sys


def _say(msg: str) -> None:
    if os.environ.get("B2S_LAUNCH_QUIET") != "1":
        print(f"[b2s-launch] {msg}", flush=True)


TP_SPEED_ENV = {"NCCL_NCHANNELS_PER_PEER": "8", "TORCH_NCCL_HIGH_PRIORITY": "1"}
TP_SPEED_PATCH = "triattn_tiles"


def _n_gpu(argv) -> int:
    """`--n_gpu P` / `--n_gpu=P` of a `pred` command line (1 when absent or not a pred)."""
    args = list(argv or [])
    if not args or args[0] != "pred":
        return 1
    for i, a in enumerate(args):
        if a == "--n_gpu" and i + 1 < len(args):
            v = args[i + 1]
        elif a.startswith("--n_gpu="):
            v = a.split("=", 1)[1]
        else:
            continue
        try:
            return int(v)
        except ValueError:
            return 1
    return 1


def tp_speed_defaults(argv) -> str:
    """The sharded line's speed words, on by default at n_gpu > 1 (B2S_TP_SPEED=0 opts out). Returns what was set ('' when nothing)."""
    if _n_gpu(argv) <= 1 or os.environ.get("B2S_TP_SPEED", "1").strip() == "0":
        return ""
    set_now = []
    for k, v in TP_SPEED_ENV.items():
        if k not in os.environ:
            os.environ[k] = v
            set_now.append(f"{k}={v}")
    patches = [p.strip() for p in os.environ.get("B2S_WORKER_PATCHES", "").split(",") if p.strip()]
    if TP_SPEED_PATCH not in patches:
        patches.append(TP_SPEED_PATCH)
        os.environ["B2S_WORKER_PATCHES"] = ",".join(patches)
        set_now.append(TP_SPEED_PATCH)
    return ",".join(set_now) or "already-set"


BIG_FLASH_ENV = {"B2S_BIG_BASE": "flash", "OF3T_FLASH_IP": "ieee", "B2S_TRIATTN_CONFIG": "128,32,1,4,1,1"}
BIG_FLASH_PATCHES = ("trimul_v4_only", "triattn_tiles")
PAIRTRACK_ONLY_LEVERS = ("pairblock", "pairblock_c64", "fused_transition", "pairfuse", "condproj")


def _mode(argv) -> str:
    """The mode of a `pred` command line as the kit resolves it: --mode, else $BOLTZ2_OPT, else the kit's default (fast)."""
    args = list(argv or [])
    for i, a in enumerate(args):
        if a == "--mode" and i + 1 < len(args):
            return args[i + 1]
        if a.startswith("--mode="):
            return a.split("=", 1)[1]
    return os.environ.get("BOLTZ2_OPT", "").strip() or "fast"


def big_flash_defaults(argv) -> str:
    """big's one-GCD flash recipe, on by default for `pred --mode big` on one GCD (B2S_BIG_FLASH=0, or an explicit B2S_BIG_BASE other
    than flash, leaves the shipped base). Returns what was set ('' when nothing)."""
    args = list(argv or [])
    if not args or args[0] != "pred" or _mode(args) != "big" or _n_gpu(args) > 1:
        return ""
    if os.environ.get("B2S_BIG_FLASH", "1").strip() == "0":
        return ""
    if os.environ.get("B2S_BIG_BASE", "flash").strip().lower() != "flash":
        return ""
    set_now = []
    for k, v in BIG_FLASH_ENV.items():
        if k not in os.environ:
            os.environ[k] = v
            set_now.append(f"{k}={v}")
    patches = [p.strip() for p in os.environ.get("B2S_WORKER_PATCHES", "").split(",") if p.strip()]
    for p in BIG_FLASH_PATCHES:
        if p not in patches:
            patches.append(p)
            set_now.append(p)
    os.environ["B2S_WORKER_PATCHES"] = ",".join(patches)
    return ",".join(set_now) or "already-set"


def _compose_big_row(M, base: str) -> dict:
    """`big`'s row as the kit's own modes.py composes it with BIG_ATTN_BASE = <base>: the module's source executed once more in a private
    namespace with that one constant changed (modes.py imports only the standard library and .registry). Nothing on disk changes."""
    path = M.__file__
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    old = f'BIG_ATTN_BASE = "{M.BIG_ATTN_BASE}"'
    if src.count(old) != 1:
        raise SystemExit(f"[b2s-launch] B2S_BIG_BASE: {old!r} is not in {path} exactly once")
    ns = {"__name__": f"boltz2_opt._b2s_modes_{base}", "__package__": "boltz2_opt", "__file__": path}
    exec(compile(src.replace(old, f'BIG_ATTN_BASE = "{base}"', 1), path, "exec"), ns)
    return ns["MODES"]["big"]


def apply() -> dict:
    """Patch boltz2_opt.modes in this process; returns what was patched (also printed)."""
    from boltz2_opt import modes as M        # standard library + boltz2_opt.registry only: nothing of opt_core or torch is imported here
    done = {"kernels": None, "card_drops": []}
    base = os.environ.get("B2S_BIG_BASE", "").strip().lower()
    if base:                                 # first: the card drops below are checked against the row's levers
        if base not in M.BIG_ATTN_BASES:
            raise SystemExit(f"[b2s-launch] B2S_BIG_BASE must be one of {','.join(M.BIG_ATTN_BASES)}, not {base!r}")
        if base != M.BIG_ATTN_BASE:
            row = _compose_big_row(M, base)
            for key in ("levers", "env", "attach"):
                M.MODES["big"][key] = row[key]
            if "fused_transition" not in M.MODES["big"]["levers"] and "BOLTZ_TRANSITION" in M.MODES["big"]["env"]:
                # modes.py sets _BIG_ROW["BOLTZ_TRANSITION"] = "big" AFTER composing either base, so the flash row carries the fused
                # transition's switch without its lever or its `transition` attachment; nothing on that row reads it (the WASTE Transition
                # patch sees the stock forward and is skipped under BOLTZ_XL), but the exit gate then demands a transition_report and exits 3
                # (speed1b 50480198). The word leaves with its lever, as modes.take_off does.
                M.MODES["big"]["env"] = {k: v for k, v in M.MODES["big"]["env"].items() if k != "BOLTZ_TRANSITION"}
            M.BIG_ATTN_BASE = base
        done["big_base"] = base
    kern = os.environ.get("B2S_KERNELS", "").strip().lower()
    if kern:
        if kern not in ("on", "off"):
            raise SystemExit(f"[b2s-launch] B2S_KERNELS must be on|off, not {kern!r}")
        for m in ("exact", "fast", "big"):
            M.MODES[m]["kernels"] = kern
        done["kernels"] = kern
    spec = os.environ.get("B2S_CARD_DROPS", "").strip()
    if spec:
        cc = os.environ.get("B2S_SHIM_CC", "9.0").strip() or "9.0"
        for item in [x.strip() for x in spec.split(";") if x.strip()]:
            parts = item.split(":", 2)
            if len(parts) != 3 or not all(parts):
                raise SystemExit(f"[b2s-launch] B2S_CARD_DROPS entry {item!r} is not <mode>:<lever>:<reason>")
            mode, lever, reason = (p.strip() for p in parts)
            if mode not in M.MODES or mode == "off":
                raise SystemExit(f"[b2s-launch] B2S_CARD_DROPS: unknown worker mode {mode!r}")
            if lever not in M.MODES[mode]["levers"]:
                if mode == "big" and done.get("big_base") == "flash" and lever in PAIRTRACK_ONLY_LEVERS:
                    done.setdefault("skipped_drops", []).append(lever)     # the flash row does not carry the pair track's levers
                    continue
                raise SystemExit(f"[b2s-launch] B2S_CARD_DROPS: {lever!r} is not a lever of mode {mode!r} ({','.join(M.MODES[mode]['levers'])})")
            M.CARD_DROPS.setdefault(cc, {}).setdefault(mode, {})[lever] = {"reason": reason}
            done["card_drops"].append(f"{mode}:{lever}:{reason}@cc{cc}")
    wp = os.environ.get("B2S_WORKER_PATCHES", "").strip()
    if wp:
        from boltz2_opt import stack as S      # standard library + the kit's own modules (stack imports no torch at module level)
        boot = os.path.join(os.path.dirname(os.path.abspath(__file__)), "b2s_worker_boot.py")
        if not os.path.isfile(boot):
            raise SystemExit(f"[b2s-launch] B2S_WORKER_PATCHES set but {boot} is missing")
        orig_cmd = S.worker_command

        def worker_command(*a, **kw):
            cmd = list(orig_cmd(*a, **kw))
            if cmd[:3] == [sys.executable, "-m", "boltz2_opt.worker_launch"]:
                return [sys.executable, boot] + cmd[1:]                    # boot -m boltz2_opt.worker_launch <the kit's args>
            _say(f"worker command not under boltz2_opt.worker_launch, left unpatched: {' '.join(cmd[:3])} ...")
            return cmd                                                     # (every worker mode of this kit has launch options)

        S.worker_command = worker_command
        done["worker_patches"] = wp
    _say(f"kernels={done['kernels'] or 'as-shipped'} card_drops={','.join(done['card_drops']) or '-'} worker_patches={done.get('worker_patches') or '-'}"
         + (f" big_base={done['big_base']}" if done.get("big_base") else "")
         + (f" skipped_drops={','.join(done['skipped_drops'])}(not_in_flash_row)" if done.get("skipped_drops") else ""))
    return done


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    tp = tp_speed_defaults(args)                                       # before apply(): both add worker-patch words / env words
    if tp:
        _say(f"tp_speed=on ({tp}; B2S_TP_SPEED=0 opts out)")
    bf = big_flash_defaults(args)
    if bf:
        _say(f"big_flash=on ({bf}; B2S_BIG_FLASH=0 opts out)")
    apply()
    from boltz2_opt.__main__ import main as kit_main     # the kit's own entry: core pin gate, module check, then cli.main
    return kit_main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
