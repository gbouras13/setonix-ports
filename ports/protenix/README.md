# protenix: Protenix v2 on Setonix, from one GCD to a whole node

Protenix v2 (`protenix` 2.0.0, ByteDance) on Setonix's MI250X GCDs, through the optimisation kit's `big` line, with the recipe that
makes it run on gfx90a. The evidence is in `af3-setonix/PROTENIX_PORT.md`.

**Status: not yet validated on Setonix** (`slurm/validate.sbatch` compares this image with the native port, byte for byte).

What the native port measured:

| line | what it is | against stock | capacity (1 cycle, 10 steps, 5 samples) |
|---|---|---|---|
| `stock` | stock Protenix's own CLI | — | — |
| `pred`, one GCD | the kit's `big` line: stock's arithmetic with less memory | **byte-identical** under deterministic algorithms (1BRS, jobs 50030425, 50379609) | up to 3,584 tokens, but not 3,072 (the kit's fused transition takes its whole projection there); 4,096 does not fit |
| `pred --n_gpu 8` | `big`, row-sharded over the node's 8 GCDs | pair stack within stock's error class against fp64; a different summation order, so not bitwise; DiT attention below | **12,288 tokens completed** (job 50122549, before the fused triangle attention became the default); 14,336 does not fit |

- On the sharded line, the trunk and confidence head run opt_core's fused triangle attention: 3.1x faster on a 6,144-token run on
  8 GCDs (job 50350451). Scaled from that run, a default prediction at 12,288 tokens would take about 6.5 h (an estimate).
- The sharded line's DiT attention is about 100x stock's fp32 denoiser error at high noise (the engine statement, the default).
  `ROWPAIR_DIFF_BIAS_CACHE_DTYPE=fp32` restores stock's precision. It is free up to about 12,000 tokens on 8 GCDs, then costs 8.7x per
  sampling step (job 50072061).

## What makes it run on gfx90a

None of these is an edit to a kit or Protenix file. `scripts/ptx-env.sh` (verbatim from the native port) and `scripts/ptx-pred.sh` set
them, each with its reason:

1. **The kit's GPU gate.** `PROTENIX_OPT_FORCE=1`: the kit identifies the GPU with `nvidia-smi` alone. `LAYERNORM_TYPE=torch`: stock
   otherwise JIT-builds a CUDA LayerNorm with `nvcc`.
2. **Four levers dropped by name.** `fastln_prebuilt`, `ln_core` and `sampler_fuse` cannot run here. `transition_core` builds and then
   returns NaN with exit 0 (PTX inline assembly), so check a lever's output before trusting it on this card.
3. **`HIP_LAUNCH_BLOCKING=1`** (rocBLAS's gfx90a gap).
4. **One GCD: `expandable_segments:False`.** The kit's default strands freed memory on ROCm 7.2. Half of the one-GCD ladder failed on it,
   and all of those runs complete without it (jobs 50229219, 50233809).
5. **`--n_gpu P`: the speed and precision settings.** The engine's DiT attention, opt_core's fused triangle attention at IEEE precision
   with the OpenFold3 port's gfx90a tiles, two RCCL settings (bitwise), and stock's triangle-attention chunk capped at 4 rows.

## Using the image

```bash
# 1. pull it (a compute job; Pawsey asks for pulls there)
sbatch --account=<project> slurm/pull_image.sbatch ghcr.io/gbouras13/setonix-ports/protenix:main /scratch/<project>/$USER/singularity

# 2. the weights, once (in a compute job): 2.5 GB, each file checked against the kit's pinned sha256; files already present are kept
module load singularity/4.1.0-nohost
singularity exec $SIF protenix fetch-weights /scratch/<project>/$USER/protenix-weights

# 3. in a GPU job (never --rocm)
singularity exec $SIF protenix check --weights $WEIGHTS --predict             # GPUs, libraries, and stock vs one GCD byte for byte
singularity exec $SIF protenix pred --weights $WEIGHTS -i complex.json -o out  # one GCD
singularity exec $SIF protenix pred --weights $WEIGHTS --n_gpu 8 -i large.json -o out --use_msa false   # a whole node
singularity exec $SIF protenix stock --weights $WEIGHTS -i complex.json -o out # stock Protenix
singularity exec $SIF protenix compare out_a out_b                             # bytes, CA RMSD, confidences
```

- Inputs are Protenix JSON, and stock's options pass through (`--seeds`, `--cycle`, `--step`, `--sample`, `--use_msa`). The launcher
  adds `--model_name protenix-v2 --triatt_kernel torch --trimul_kernel torch` unless you give them.
- The weights: upstream's own checkpoint URL answers HTTP 403. `fetch-weights` takes the checkpoint from a Hugging Face mirror
  (`TMF001/protenix-v2-weights`), as the native port did, and keeps it only if its sha256 is the kit's pin. The six data caches come
  from upstream through the kit's own routine. A checkpoint you already have can go to `DIR/checkpoint/protenix-v2.pt` first.
- One GCD above about 2,500 tokens: add `PXS_TRIATT_CHUNK_CAP=4`, as the native 3,584-token run had. It bounds stock's
  triangle-attention logits and changes only memory and speed.
- `--n_gpu P` puts its ranks on the first P GCDs the process sees, so give concurrent runs their own `ROCR_VISIBLE_DEVICES`.
- Setonix starts every job from a fresh login environment (`SBATCH_EXPORT=NONE`). Set `PROTENIX_ROOT_DIR` (or use `--weights`) and any
  `PXS_*` / `ROWPAIR_*` setting inside the job script.
- Deterministic runs: `PXS_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8`, with `stock` rather than `pred --mode off`, which refuses
  the deterministic hook.
- `protenix kit ...` runs the kit's own CLI (`python -m protenix_opt`) in the port's environment, without the recipe.

## Files

| path | what |
|---|---|
| `container/Dockerfile` | the image: python-build-standalone 3.12.12, the lock, the kit at `f4f62fa6`, the hooks and scripts; torch's ROCm kernels pruned to gfx90a |
| `container/protenix`, `container/protenix_check.py` | the entry point and its self-check |
| `environment/requirements.lock` | the 48 packages the native port runs on, at its versions (its venv is layered on another env: this is the closure) |
| `hooks/` | the native port's start-up hooks, verbatim: per-rank caches, the chunk cap, deterministic runs, two diagnostics |
| `scripts/ptx-env.sh`, `scripts/ptx-pred.sh` | the native environment, verbatim, and the launcher that picks the recipe for a route |
| `tools/` | output comparison (`ptx_outputs.py`), the native deterministic comparison (`compare_det.py`) and checkpoint fetch, verbatim |
| `tests/test_launcher.sh` | dry-run tests of the launcher's decisions (run by `checks.yml`) |
| `slurm/validate.sbatch` | on one node: the weights fetch, the check, and every line byte for byte against the native port's stored runs |
