# boltz2: Boltz-2 on Setonix, every kit mode, from one GCD to a whole node

Boltz-2 (`boltz` 2.2.1) on Setonix's MI250X GCDs, through the optimisation kit's four modes, with the recipes that make each one run on
gfx90a. The evidence is in `af3-setonix/BOLTZ2_PORT.md`.

**Status: not yet validated on Setonix** (`slurm/validate.sbatch` compares this image with the native port, byte for byte).

On the HK97 capsid trimer with its ColabFold MSA (1,155 tokens, seed 1, upstream settings: 3 recycles, 200 steps), against stock in the
same job:

| mode | what it is | speed vs stock | against stock |
|---|---|---|---|
| `off` | stock `boltz predict`, in the kit's clean subprocess | 1.00x (279 s) | — |
| `exact` | the kit's exact mode: a graphed sampler roll-out, kernels that keep stock's arithmetic | 1.11-1.13x | **byte-identical** |
| `fast` | the kit's default mode, with opt_core's flash triangle attention in its fused block | **2.1x**; **2.9x** with `B2S_FAST_PATCH=1` | 1.7 Å CA RMSD, on stock's own arrangement; 0.13 Å from `big` |
| `big`, one GCD | the kit's memory mode on its flash base (opt_core's flash triangle attention) | **2.5-2.7x** | within 0.06-0.15 Å of the kit's shipped big |
| `big --n_gpu 2` | row-sharded over two GCDs of the node | **2.2x** | 0.38 Å |

Stock's own seeds land 14-35 Å apart on this input, so every mode sits well inside the model's sampling spread. The flash kernel is more
precise than stock's own triangle attention (0.83-0.90x its error against float64, on every one of 20 live calls). On a low-confidence
input the flash and sharded lines can still land on a different structure for a given seed, as a different seed would.

Capacity at 1 recycle and 10 sampling steps (HK97 k-mers with the MSA):

| line | fits | out of memory |
|---|---|---|
| `off` / `exact` / `fast`, one GCD | 2,310 tokens | 2,695 |
| `big`, one GCD | 2,695 | 3,080 |
| `big --n_gpu 2` | 8,085 | 8,470 |
| `big --n_gpu 4` | 10,395 | 11,550 |
| `big --n_gpu 8` | **15,785** | 16,170 |

Above ~12,000 tokens, `--n_gpu 8` needs a gpu-highmem node for host memory (323-332 GiB at 15,400-16,170). Run **one sharded prediction
per node**: two on the same node slowed each other up to ~70x.

## What makes it run on gfx90a

None of these is an edit to a kit file (`af3-setonix/BOLTZ2_PORT.md` has the measurements):

1. **Kernels off, levers dropped by name.** No cuEquivariance on ROCm, so every mode runs `B2S_KERNELS=off` (`scripts/b2s_launch.py`
   patches the kit's mode table in the parent). The levers that cannot run here leave through the kit's own card-drop table, each with
   the reason seen (`scripts/b2s-recipes.sh`): NVRTC / `cuda.bindings` units, two H100 tile configs above gfx90a's 64 KB of LDS, two
   torch-pinned units. A worker-side boot patch (`scripts/b2s_worker_boot.py`) refuses two opt_core kernel rows by name: the
   transition's `v2` (all-NaN here) and every TriMul row but `v4` (the next one carries NVPTX inline assembly).
2. **`HIP_LAUNCH_BLOCKING=1`** (`scripts/bz-env.sh`). Without it stock Boltz-2 itself dies above ~500 tokens: a SIGSEGV in a different
   host frame every time, or a hang at exit. With it, everything completes, and outputs are byte-identical to runs that happened to
   finish without it.
3. **`DEBUG_CLR_GRAPH_PACKET_CAPTURE=0`** (`scripts/bz-env.sh`). ROCm 7.2 corrupts the host heap replaying a captured graph; fast's
   graphed sampler roll-out dies without it. See `af3-setonix/rocm-graph-bug/REPORT_AMD.md`.
4. **The kit's flash triangle attention, retuned.** One-GCD big runs the kit's flash base with ROWS=1 tiles (the ROWS>=2 mask scan does
   not compile on gfx90a / Triton 3.8); `--n_gpu P` gets OpenFold3's tiles and RCCL words. Both are `scripts/b2s_launch.py` defaults
   (`B2S_BIG_FLASH=0` / `B2S_TP_SPEED=0` opt out).
5. **fast's fused block, allowed to serve.** The kit's fused triangle-attention block serves only when boltz asks for kernels, and the
   port must run kernels off, so fast ran stock's chunked attention. `scripts/b2s_launch.py` puts the block on its `flash` variant
   (opt_core's own kernel, no cuEquivariance), and the boot word `pairblock_kernels` lets it serve under kernels off. Every call it
   declines runs stock's code exactly as before. `B2S_FAST_FLASH=0` opts out.

## Using the image

```bash
# 1. pull it (a compute job; Pawsey asks for pulls there)
sbatch --account=<project> slurm/pull_image.sbatch ghcr.io/gbouras13/setonix-ports/boltz2:main /scratch/<project>/$USER/singularity

# 2. the weights, once: 6.55 GB from Hugging Face (boltz-community, MIT), each file checked against the kit's pinned sha256
#    (in a compute job; a file already present with the right digest is kept)
module load singularity/4.1.0-nohost
singularity exec $SIF boltz2 fetch-weights /scratch/<project>/$USER/boltz2-weights

# 3. in a GPU job (never --rocm)
singularity exec $SIF boltz2 check --weights $WEIGHTS --predict      # GPUs, libraries, and stock vs exact byte for byte
singularity exec $SIF boltz2 pred --weights $WEIGHTS --mode big --input complex.yaml --out_dir out
singularity exec $SIF boltz2 pred --weights $WEIGHTS --mode big --n_gpu 8 --input large.yaml --out_dir out      # a whole node
singularity exec $SIF boltz2 pred --weights $WEIGHTS --mode exact --input complex.yaml --out_dir out --seed 1  # = stock, faster
singularity exec $SIF boltz2 compare out_a out_b                                                        # bytes, CA RMSD, scores
```

- Inputs are Boltz YAML. A ColabFold a3m ends each query block with a NUL byte that Boltz's parser rejects: `tr -d '\000'` it first.
- Each `pred` stages the weights on node-local disk (`mols/` unpacked, ~45,000 files) under `B2S_LOCAL`, by default a fresh directory
  under `TMPDIR`. Concurrent predictions in one job each get their own; give each its own `ROCR_VISIBLE_DEVICES`.
- Setonix starts every job from a fresh login environment (`SBATCH_EXPORT=NONE`): set `B2S_WEIGHTS`, `B2S_FAST_PATCH` and friends
  inside the job script, or use `--weights`.
- `boltz2 kit ...` runs the kit's own `run.sh`, unpatched.

## Files

| path | what |
|---|---|
| `container/Dockerfile` | the image: python-build-standalone 3.12.12, the lock, the kit at `f4f62fa6`, the scripts; torch's ROCm kernels pruned to gfx90a |
| `container/boltz2`, `container/boltz2_check.py` | the entry point and its self-check |
| `environment/requirements.lock` | the 59 packages the native port runs on, at its versions (its venv is layered on another env: this is the closure) |
| `scripts/` | the native port's scripts, verbatim: environment, launcher, boot patch, recipes, weights fetch and stage, nvidia-smi shim; `b2s-pred.sh` picks the recipe for a mode |
| `tools/b2s_outputs.py` | output summary and comparison |
| `slurm/validate.sbatch` | the image against the native port on one node: check, every mode byte for byte, the weights fetch |
