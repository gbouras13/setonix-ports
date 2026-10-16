"""Deterministic algorithms and prediction dumps for Protenix runs on Setonix (env-gated, diagnostics).

PXS_DETERMINISTIC=1   torch.use_deterministic_algorithms(True, warn_only=True) at interpreter start, in every process that loads
                      the venv (the stock CLI, the kit's env route, and every rank of the kit's torchrun line). Later calls to
                      torch.use_deterministic_algorithms are logged; one that would turn the mode off is logged and kept on (this
                      run's premise). The kit's own --det 1 sets the same flag (src/sitecustomize.py, PTX_DET=1) and also copies a
                      deterministic scatter into the installed package; torch's deterministic mode already makes stock's only
                      nondeterministic trunk op (the atom-to-token scatter_add_) repeat bitwise (job 49996035), so this needs no
                      package edit. CUBLAS_WORKSPACE_CONFIG must be in the job's environment before CUDA starts.
PXS_DUMP_PRED=<dir>   runner.inference.InferenceRunner.predict wrapped (through the kit's opt_core.autoload.patch_attr_at_import):
                      every prediction's tensors (coordinate, plddt / pae / pde / resolved logits, contact_probs) and summary
                      confidences saved as <dir>/pred_r<rank>_<n>.pt, for bitwise comparison across runs. Not with the kit's
                      `--mode off`: its stock-purity check refuses a process with an opt_core.autoload patch ("NOT STOCK: autoload
                      ['AttrPatch']", job 50018899), and in that job no kit-mode dump fired either.
At exit: one line with the deterministic state and the logged calls.
"""
import atexit
import os
import sys

_LOG = {"calls": [], "kept_on": 0, "dumps": 0}
_RANK = os.environ.get("RANK") or os.environ.get("ROWPAIR_RANK") or "0"


def _say(msg):
    print(f"[pxs-det r{_RANK}] {msg}", file=sys.stderr, flush=True)


def arm_det():
    import torch
    orig = torch.use_deterministic_algorithms
    orig(True, warn_only=True)

    def use_deterministic_algorithms(mode, *, warn_only=False):
        caller = sys._getframe(1)
        where = f"{caller.f_code.co_filename}:{caller.f_lineno}"
        _LOG["calls"].append((bool(mode), bool(warn_only), where))
        if not mode:
            _LOG["kept_on"] += 1
            _say(f"use_deterministic_algorithms(False) from {where}: kept ON (PXS_DETERMINISTIC=1)")
            return orig(True, warn_only=True)
        return orig(True, warn_only=True)
    torch.use_deterministic_algorithms = use_deterministic_algorithms
    _say(f"torch.use_deterministic_algorithms(True, warn_only=True); CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}")


def arm_dump(outdir):
    from opt_core.autoload import patch_attr_at_import
    os.makedirs(outdir, exist_ok=True)

    def make(orig):
        def predict(self, data):
            out = orig(self, data)
            try:
                import torch
                keep = {k: v.detach().to("cpu").clone() for k, v in out.items() if torch.is_tensor(v)}
                summ = []
                for s in out.get("summary_confidence", []) or []:
                    summ.append({k: (v.detach().to("cpu").clone() if torch.is_tensor(v) else v) for k, v in s.items()})
                keep["summary"] = summ
                keep["deterministic"] = bool(torch.are_deterministic_algorithms_enabled())
                p = os.path.join(outdir, f"pred_r{_RANK}_{_LOG['dumps']}.pt")
                torch.save(keep, p)
                _LOG["dumps"] += 1
                _say(f"prediction dumped: {p} (deterministic={keep['deterministic']}, keys={sorted(k for k in keep if k != 'summary')})")
            except Exception as e:  # noqa: BLE001 — a dump must never cost the prediction
                _say(f"dump failed: {e!r}")
            return out
        return predict

    patch_attr_at_import("runner.inference", "InferenceRunner.predict", make, tag="pxs-dump", name="pxs:dump_pred")


def _exit():
    try:
        import torch
        on = bool(torch.are_deterministic_algorithms_enabled())
    except Exception:  # noqa: BLE001
        on = None
    _say(f"exit: deterministic={on} calls_logged={len(_LOG['calls'])} kept_on={_LOG['kept_on']} dumps={_LOG['dumps']} "
         f"calls={_LOG['calls'][:6]}")


def arm():
    if os.environ.get("PXS_DETERMINISTIC") == "1":
        arm_det()
    if os.environ.get("PXS_DUMP_PRED"):
        arm_dump(os.environ["PXS_DUMP_PRED"])
    atexit.register(_exit)
