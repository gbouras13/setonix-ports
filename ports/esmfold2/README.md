# esmfold2: ESMFold2 on Setonix, `exact` byte-identical to stock

ESMFold2 and ESMFold2-Fast (Biohub `esm` 3.3.0 with its `transformers` fork 4.57.6, and the ESMC-6B language model) on Setonix's MI250X
GCDs, through the optimisation kit's modes, with the fixes that make them run on gfx90a. The evidence is in `af3-setonix/ESMFOLD2_PORT.md`.

- **Validated on Setonix** (2026-10-21), image `esmfold2-image` (source 61cc14a), digest `sha256:2a943740…` (job 51303129, every section
  passes):
  - `esmfold2 check --predict` on all 8 GCDs: the tree, the kit's pin check, the weights through the kit's own sha256 check, every GPU
    library loaded from inside the image, and 1BRS (199 tokens) through stock-fused and `exact` inside the image, byte-identical.
  - The 48 bench targets (and the 11 MSA inputs) through every route in the image, `--det 1`: `exact`, `fast`, stock-fused on
    ESMFold2-Fast and on ESMFold2, `exact` with MSAs, `big` — byte-identical to the native port (job valid6 51248511; 48/48 on each,
    11/11 with MSAs), at native speed (`exact` 619 s against 654, `fast` 513 against 489, stock-fused 889 against 946, `big` 1,361
    against 1,403).
  - `esmfold2 fetch-weights`: the 16 files from Hugging Face (27.5 GB in 22 minutes), the kit's sha256 check, and identical to the
    native copies.

What the native port measured (48 recent PDB monomers, 105-497 residues; one process per GCD, all eight GCDs of a node busy; library
defaults: 20 recycles, 200 sampling steps, 1 sample; job valid6 51248511):

| route | what it is | against stock | ESMFold2-Fast, per node | ESMFold2, per node |
|---|---|---|---|---|
| `stock` | upstream as shipped (`--mode off`) | — | 718/h (`--det 0`) | 382/h (`--det 1`) |
| `stock-fused` | upstream's fused Triton backend | — | 1,462/h | 971/h |
| **`exact`** | the kit's exact mode | **byte-identical to `stock-fused`** on every variant (48/48, 48/48, 11/11 with MSAs) | **2,114/h** (1.45x) | **1,345/h** (1.38x) |
| `fast` | the kit's fast mode, `tx` off | median CA RMSD 0.041 A to stock-fused, max 1.07 A (stock shipped vs fused: 0.015-0.027, max 3.5-7.8) | 2,827/h (1.93x) | 1,632/h (1.68x) |
| `big` | the memory mode; `--n_gpu P` row-shards one input | median 0.051 A, max 5.7 A | 985/h | |

Accuracy against the deposited structures is the same on every route (US-align TM-score 0.958 mean, 0.981 median for ESMFold2-Fast).

Longest input (3 recycles, 50 steps): one GCD 1,800 tokens on `exact` / `fast` (1,900 with `EFS_ALLOC=True`, one input per process),
1,800 on `big`, 1,800 on stock shipped, 1,600 on stock fused; `big --n_gpu 2` 4,500 tokens, `--n_gpu 4` 6,500, `--n_gpu 8` below 9,000 (the kit's allocator holds memory back on ROCm 7.2: P=8 buys nothing over P=4).

## What makes it run on gfx90a

None of these is an edit to a kit or upstream file. `scripts/ef2.sh` (verbatim from the native port) sets them per route:

1. **Two boot fixes, in memory, per run** (`scripts/boot/sitecustomize.py`, `EFS_BOOT`). `dotprec`: upstream's fused pair-bias kernel
   asks Triton for `tf32x3`, which the AMD backend rejects, so `--backend fused` cannot compile; the word becomes `ieee` (true fp32,
   native on MI250X). `smem`: torch on ROCm has no `shared_memory_per_block_optin`, so the kit assumed an H100's 227 KB from compute
   capability 9.0 (what gfx90a reports); the fix reports the card's 64 KB.
2. **The kit's own ablation switch** (`MODEL_OPT_LEVERS_OFF`) for the levers its 64 KB policy then refuses: `t3,t6` for `exact` (and
   `glue`, whose kernel fails the kit's own sigmoid check here, on the full model), `t3,t16,t10` for `fast` / `big`, and `tx` on `fast`:
   without it `fast` writes NaN for every input above 256 tokens.
3. **`HIP_LAUNCH_BLOCKING=1`** on every route: ROCm 7.2 crashed every route asynchronously without it.
4. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`** on `exact` / `fast`: with the kit's expandable segments, memory torch unmaps
   is never given back on ROCm 7.2, and a process folding many inputs ran out after 39-48 of them. Without, memory stays flat and the
   bytes are the same.
5. **The kit's `run.sh`** refuses this stack at its image-layer pin check (the CUDA accelerated layer: flash-attn, transformer-engine,
   xformers). `scripts/ef2-run.sh` runs a copy with exactly two lines changed and checks that nothing else differs; the forks' commit
   pins still refuse. Upstream runs its own fallbacks without that layer (SDPA atom attention, PyTorch LayerNorm, MLP and rotary).

## Using the image

```bash
# 1. pull it (a compute job; Pawsey asks for pulls there)
sbatch --account=<project> slurm/pull_image.sbatch ghcr.io/gbouras13/setonix-ports/esmfold2:main /scratch/<project>/$USER/singularity

# 2. the weights, once (in a compute job): 27.5 GB, every file checked by the kit against its pinned sha256; files already there stay
module load singularity/4.1.0-nohost
singularity exec $SIF esmfold2 fetch-weights /scratch/<project>/$USER/esmfold2-weights

# 3. in a GPU job (never --rocm)
export EFS_WEIGHTS=/scratch/<project>/$USER/esmfold2-weights EFS_CACHE=/scratch/<project>/$USER/esmfold2-cache
singularity exec $SIF esmfold2 check --predict                     # GPUs, libraries, and stock-fused vs exact byte for byte
singularity exec $SIF esmfold2 stage                               # the weights to node-local disk, once per job
for g in 0 1 2 3 4 5 6 7; do                                       # one process per GCD
  ROCR_VISIBLE_DEVICES=$g singularity exec $SIF esmfold2 pred exact --variant fast --input part$g.json --out_dir out$g &
done; wait
singularity exec $SIF esmfold2 pred big --variant fast --n_gpu 4 --input long.json --out_dir out_long   # one input on 4 GCDs
singularity exec $SIF esmfold2 unstage                             # free the node's /tmp
```

- Inputs are upstream's JSON: `{"id": ..., "sequences": [{"type": "protein", "id": "A", "sequence": "..."}]}`, one item or a list;
  a chain's `"msa"` may name an A3M file (`--variant full_msa`). Variants: `fast` (ESMFold2-Fast), `full_nomsa`, `full_msa` (ESMFold2).
- Outputs: `OUT/cif_all/<id>__<fast|full>__s<seed>_x<k>.cif` with `_pae.npz` (PAE, pLDDT), and `OUT/pred_rows.jsonl`.
- Any number of inputs per process. The first run of a kit route compiles Triton kernels (minutes): keep `EFS_CACHE` on /scratch and
  later jobs reuse them.
- `--det 1` turns on the kit's deterministic recipe (what the byte comparisons use). The kit's other `pred` options pass through
  (`--seeds`, `--num_loops`, `--num_sampling_steps`).
- Setonix starts every job from a fresh login environment (`SBATCH_EXPORT=NONE`): set `EFS_*` inside the job script.
- `esmfold2 kit ...` runs the kit's own `run.sh` (through the two-line copy) in the port's environment, without the route's fixes.

## Files

| path | what |
|---|---|
| `container/Dockerfile` | the image: python-build-standalone 3.12.12, the lock, the kit at `f4f62fa6`; torch's ROCm kernels pruned to gfx90a |
| `container/esmfold2`, `container/esmfold2_check.py` | the entry point and its self-check |
| `environment/requirements.lock` | the native venv's freeze, line for line (the build fails if the image's differs) |
| `scripts/` | the native port's environment, routes (`ef2.sh`), `run.sh` wrapper and pin check, weights stage, boot fixes and `nvidia-smi` shim, verbatim; `fetch-weights.sh` |
| `tools/ef2_compare.py` | output comparison: byte identity, then coordinates, PAE / pLDDT and rows |
| `inputs/smoke_1brs.json` | 1BRS barstar + barnase (199 tokens), for `esmfold2 check --predict` |
| `slurm/validate.sbatch` | on one node: the check, every route byte for byte against the native port's stored runs, and the weights fetch |
