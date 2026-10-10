#!/usr/bin/env python
"""pms_dp.py -- one design pass over several GCDs of a node, data-parallel, for ProteinMPNN (stock or the kit's exact worker).

    python pms_dp.py --mode exact|off --input <dir of PDBs | parsed.jsonl> --out OUT --gcds 0,1,2,3,4,5,6,7 [--per-gcd K]
                     [--bb_batch N] [--variant vanilla|soluble] [protein_mpnn_run.py options: --seed 37 --num_seq_per_target 8 ...]

mode exact: the kit's own exact worker, staged by the kit's own stage.stage_base(), with the kit's own lever line for `--mode exact
            --hybrid_gemm 0` (modes.resolve) and the stock options rendered by the kit's own settings.kit_argv(). Each of the G x K
            workers runs on one GCD (ROCR_VISIBLE_DEVICES) through scripts/pms_worker_boot.py with the boot patches of PMS_WORKER_PATCHES
            (default rocm_draw) plus `shard=i/n`: every worker reads the WHOLE parsed jsonl and computes every backbone's offset in
            the one-process stock RNG stream (the kit's stream replay), then designs only the batches i, i+n, i+2n, ... of the kit's own
            length-sorted batch plan. So the union of the n workers' files is the single-process exact run's, which is byte-identical
            to one stock process over the same input. All workers write into OUT (distinct files).
mode off:   upstream's protein_mpnn_run.py, one process per worker on a shard of the parsed jsonl (length-sorted, round-robin). This is
            how stock is parallelised without the kit: every shard restarts the seed's stream, so the outputs are valid samples but
            NOT the single-process stock run's bytes.
The parse step (a PDB directory) is the kit's read-once parser for exact and upstream's helper for off, run once. Prints one line per
worker (`DP-WORKER i gcd rc wall_s`) and a summary `DP-SUMMARY mode=.. workers=.. inputs=.. fa=.. designs=.. wall_s=.. designs_per_s=..`;
exit 0 when every worker exited 0 and every input has its .fa.
"""
import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BOOT = os.path.join(HERE, "pms_worker_boot.py")


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--mode", required=True, choices=["exact", "off"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gcds", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--per-gcd", type=int, default=1)
    ap.add_argument("--bb_batch", type=int, default=None)
    ap.add_argument("--variant", default="vanilla")
    ap.add_argument("--keep-stage", action="store_true")
    ap.add_argument("--hybrid", action="store_true", help="request the kit's probe-gated --hybrid_gemm (the exact line is --hybrid_gemm 0 on gfx90a: its probe fails)")
    ap.add_argument("--prefix", default="", help="a command prefixed to every worker's command line (e.g. 'gdb -q -batch -ex run -ex bt --args')")
    a, stock_args = ap.parse_known_args()
    from proteinmpnn_opt import core_gate
    core_gate()
    from proteinmpnn_opt import modes, settings, stack, stage
    mdir = stack.mpnn_dir()
    gcds = [g for g in a.gcds.split(",") if g != ""]
    n = len(gcds) * a.per_gcd
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    py = sys.executable
    t0 = time.time()
    stage_dir = tempfile.mkdtemp(prefix="pms_dp_stage_")
    try:
        pairs = settings.parse(stock_args, a.variant)
        # ---- inputs: one parse ----
        if os.path.isdir(a.input):
            parsed = os.path.join(out, "parsed.jsonl")
            if a.mode == "exact":
                paths = stage.stage_base(stage_dir)
                pcmd = [py, paths["fast_parse"], "--input_path", os.path.abspath(a.input), "--output_path", parsed]
            else:
                pcmd = [py, os.path.join(mdir, "helper_scripts", "parse_multiple_chains.py"), "--input_path", os.path.abspath(a.input), "--output_path", parsed]
            r = subprocess.run(pcmd)
            if r.returncode != 0:
                print("DP parse failed rc=%d" % r.returncode, file=sys.stderr); return 1
        else:
            parsed = os.path.abspath(a.input)
            if a.mode == "exact":
                paths = stage.stage_base(stage_dir)
        names = [json.loads(l)["name"] for l in open(parsed) if l.strip()]
        t_parse = time.time() - t0
        procs = []
        if a.mode == "exact":
            res = modes.resolve("exact", a.variant, stack.kit_home(), opt_out=[] if a.hybrid else ["hybrid_gemm"], bb_batch=a.bb_batch)
            unassigned = os.path.join(stage_dir, "unassigned.jsonl")
            with open(unassigned, "w") as fh:
                fh.write("null\n")                                     # upstream's default: every chain designed (the kit's inputs.write_unassigned)
            weights = settings.base_weights_dir(pairs, a.variant, mdir)
            base = [paths["worker_lowmem"], "--jsonl_path", parsed, "--chain_id_jsonl", unassigned, "--out_folder", out,
                    "--path_to_model_weights", weights] + settings.kit_argv(pairs, a.variant) + list(res.flags)
            patches = os.environ.get("PMS_WORKER_PATCHES", "rocm_draw")
            for i in range(n):
                env = stack.kit_env()
                env["ROCR_VISIBLE_DEVICES"] = gcds[i % len(gcds)]
                env["PMS_WORKER_PATCHES"] = ",".join(p for p in (patches, "shard=%d/%d" % (i, n)) if p and p != "none")
                log = open(os.path.join(out, "dp_worker_%d.log" % i), "w")
                procs.append((i, env["ROCR_VISIBLE_DEVICES"], subprocess.Popen(shlex.split(a.prefix) + [py, BOOT] + base, env=env, stdout=log, stderr=subprocess.STDOUT), log, time.time()))
        else:
            rows = [l for l in open(parsed) if l.strip()]
            if n > 1:                                                  # shards balanced by length; one process keeps the input's order (= one stock invocation)
                rows.sort(key=lambda l: len(json.loads(l)["seq"]))
            knobs = settings.stock_argv(pairs)
            # protein_mpnn_run.py makes its folders with check-then-makedirs: several processes sharing one output folder race on it
            # (FileExistsError). Make exactly the folders stock makes, first: seqs/ always, scores/ and probs/ when their gates are open.
            os.makedirs(os.path.join(out, "seqs"), exist_ok=True)
            for flag, sub in (("--save_score", "scores"), ("--save_probs", "probs")):
                if flag in stock_args and stock_args[stock_args.index(flag) + 1] not in ("0", ""):
                    os.makedirs(os.path.join(out, sub), exist_ok=True)
            wdir = [] if settings.weights_given(pairs) else ["--path_to_model_weights", os.path.join(mdir, "%s_model_weights" % a.variant)]
            for i in range(n):
                shard = os.path.join(out, "shard_%d.jsonl" % i)
                with open(shard, "w") as fh:
                    fh.writelines(rows[i::n])
                env = dict(os.environ)
                env["ROCR_VISIBLE_DEVICES"] = gcds[i % len(gcds)]
                log = open(os.path.join(out, "dp_worker_%d.log" % i), "w")
                cmd = [py, os.path.join(mdir, "protein_mpnn_run.py"), "--jsonl_path", shard, "--out_folder", out] + wdir + knobs
                procs.append((i, env["ROCR_VISIBLE_DEVICES"], subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log, time.time()))
        rcs = []
        for i, g, p, log, ts in procs:
            rc = p.wait(); log.close(); rcs.append(rc)
            print("DP-WORKER %d gcd=%s rc=%d wall_s=%.1f" % (i, g, rc, time.time() - ts), flush=True)
        wall = time.time() - t0
        fa = [f for f in os.listdir(os.path.join(out, "seqs"))] if os.path.isdir(os.path.join(out, "seqs")) else []
        nseq = 0
        for f in fa:
            with open(os.path.join(out, "seqs", f)) as fh:
                nseq += sum(1 for l in fh if l.startswith(">T="))
        ok = all(rc == 0 for rc in rcs) and len(fa) == len(names)
        print("DP-SUMMARY mode=%s workers=%d gcds=%s per_gcd=%d inputs=%d fa=%d designs=%d parse_s=%.1f wall_s=%.1f designs_per_s=%.2f %s" % (
            a.mode, n, ",".join(gcds), a.per_gcd, len(names), len(fa), nseq, t_parse, wall, nseq / wall if wall > 0 else 0.0, "OK" if ok else "FAILED"), flush=True)
        return 0 if ok else 1
    finally:
        if not a.keep_stage:
            shutil.rmtree(stage_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
