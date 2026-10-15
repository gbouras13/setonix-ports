# opendde: OpenDDE on Setonix, from one GCD to a whole node, to 9,216 tokens

OpenDDE (`opendde` 1.1.1) on Setonix's MI250X GCDs, through the optimisation kit's `big` line: one GCD with the kit's host offload, or
row-sharded over 2, 4 or 8 GCDs of a node by the kit's own launcher. The port adds the settings gfx90a needs and eight opt-in fixes.
The evidence is in `af3-setonix/OPENDDE_PORT_LOG.md`.

**Status: not yet validated on Setonix.** `slurm/validate.sbatch` compares this image with the native port: byte for byte on one GCD
under `--det 1`, and against the native runs on 2 and 8 GCDs.

Measured natively, 4,096 tokens on 8 GCDs (1 cycle, 40 steps, one sample):

| line | time | peak per GCD | against stock |
|---|---|---|---|
| stock (`--stock`) | 51.5 min | 22.4 GiB | — |
| patched (the default) | **21.0 min (2.45x)** | 19.3 GiB | 0.18-0.22 Å CA RMSD, inside the 0.24 Å two identical runs differ by; pLDDT identical, pTM within 0.0004 |

At the model's shipped settings (10 cycles, 200 steps, 5 samples) the same prediction takes **2 h 52 min**, at the same memory.

Capacity (synthetic protein inputs, ~8.4 atoms per token, no MSA, 1 cycle and 40 steps):

| line | largest that ran | first that did not |
|---|---|---|
| one GCD, either line | 6,144 tokens fit in memory | 8,192: out of memory in the pair stack, patched or not |
| `--n_gpu 8`, stock | 4,096, complete in 52 min (6,144 and 8,192 were still running at their time limits, in tiny attention blocks) | 10,240: out of memory (the dense-window tensors) |
| `--n_gpu 8`, patched | 8,192 complete (2 h 43 min); **9,216 complete** (3 h 50 min, with the node's whole memory: `sbatch --mem=0`) | 10,240: needs ~562 GiB of host RAM in the structural stage |

At 9,216 tokens both limits are close: host RAM peaks at 467 of the node's 478.5 GiB (the partition's default is 460: hence
`--mem=0`), and the GPU at 60.95 GiB allocated of 64 at the hand-over into diffusion. Run **one sharded prediction per node**.

## What makes it run on gfx90a

None of these is an edit to a kit or OpenDDE file.

1. **Upstream's torch triangle kernels by stated flag** (no cuEquivariance on ROCm), the sharded triangle-attention core as torch
   statements, and the four levers whose kernels cannot run here composed out by name (`dit_attn_apb`, `atom_attn_apb`, `atom_fused`,
   `cond_dedupe`: the apb sampler kernels request tf32, because the kit's capability table takes gfx90a's compute capability (9,0) for
   an H100).
2. **`HIP_LAUNCH_BLOCKING=1`.** Required on the sharded line: with 0, rank 7 segfaulted in the diffusion roll-out in 2 of 2 runs.
3. **`LAYERNORM_TYPE=torch`** (upstream's fused LayerNorm needs nvcc) and **the kit's sub-64-GiB size gates** (its card probe reads
   `nvidia-smi` only; a 64 GiB card also falls on the wrong side of its own boundary).
4. **The patched line** (`scripts/odde-pred.sh`, the native `predict_patched7.sh`): eight opt-in fixes, each installed through the
   kit's own `opt_core.autoload.patch_attr_at_import` by the hook chain in `patches/`:
   - **v2 dense-window construction**: stock OpenDDE builds two dense N_atom x N_atom tensors to carve out 32 x 128 windows (39 GiB
     at 12,288 tokens, for a 13 MB mask). Built directly, byte-identical output.
   - **explicit triangle-attention blocks** (32 rows x 256 queries): the kit takes both blocks from stock's unsharded chunk budget
     (16 x 16 at 4,096; 2 at 12,288, which it refuses). This is most of the 2.45x.
   - **`ROWPAIR_TRIMUL_SUB=128`**: the trimul sub-block table is keyed on token count only, not OpenDDE's 384-channel pair.
   - **the native allocator** (`expandable_segments:False` through the kit's own writers): its mandatory expandable segments strand
     freed memory on ROCm 7.2. The kit then marks the run `partial`, by name; rc stays 0.
   - **the A-block threshold 0.60**: the trimul planner kept a plane whole up to 0.80 of free memory (validated to ~0.72).
   - **`ROWPAIR_TRIATT_ROWBLOCK=256`**, **query-blocked stock attention** (above 4 GiB of logits) and **row-blocked APB pair bias**
     (above an 8 GiB shard): three statements in the structural refiner that each copied the whole 29.6 GiB shard at 9,216 tokens.

## Using the image

```bash
# 1. pull it (a compute job; Pawsey asks for pulls there)
sbatch --account=<project> slurm/pull_image.sbatch ghcr.io/gbouras13/setonix-ports/opendde:main /scratch/<project>/$USER/singularity

# 2. the weights, once: the checkpoint (2.6 GB, Hugging Face aurekaresearch/OpenDDE) and four common/ files (0.6 GB), through the kit's
#    own downloader, each checked against the kit's pinned sha256 (in a compute job; a file already present is kept and checked)
module load singularity/4.1.0-nohost
singularity exec $SIF opendde fetch-weights /scratch/<project>/$USER/opendde-weights

# 3. in a GPU job (never --rocm)
singularity exec $SIF opendde check --weights $WEIGHTS --predict             # GPUs, libraries, and 600 tokens byte for byte
singularity exec $SIF opendde pred --weights $WEIGHTS -i query.json -o out                     # one GCD, patched line
singularity exec $SIF opendde pred --weights $WEIGHTS --n_gpu 8 -i large.json -o out           # a whole node
singularity exec $SIF opendde pred --weights $WEIGHTS --stock --n_gpu 8 -i large.json -o out   # the kit's big line as shipped
singularity exec $SIF opendde compare a.cif b.cif                                              # CA RMSD, same sample
```

- Inputs are OpenDDE's JSON (`tests/q600.json` is an example). Everything after the port's own options goes to `python -m opendde_opt
  pred` unchanged (`--seeds`, `--sample`, `--cycle`, `--step`, `--use_msa`, `--det 1`, ...); the shipped settings apply unless given.
- Setonix starts every job from a fresh login environment (`SBATCH_EXPORT=NONE`): set `ODDE_WEIGHTS` and any `ODDE_*` override inside
  the job script, or use `--weights`. Compiled kernels and temporaries go under `ODDE_LOCAL`, by default a fresh directory under `TMPDIR`.
- Every fix is an environment word with its native default (`scripts/odde-pred.sh`); `opendde kit ...` runs the kit's own `run.sh`.

## Files

| path | what |
|---|---|
| `container/Dockerfile` | the image: python-build-standalone 3.12.12, the lock, the kit at `f4f62fa6`, the port; torch's ROCm kernels pruned to gfx90a |
| `container/opendde`, `container/opendde_check.py` | the entry point and its self-check |
| `environment/requirements.lock` | the 43 packages the native port runs on, at its versions (its venv is layered on another env: this is the closure) |
| `scripts/` | the native port's environment, its two prediction scripts as one (`odde-pred.sh`), and the per-rank cache hook |
| `patches/` | the eight opt-in fixes and the hook chain that installs them, verbatim from the native port |
| `tests/` | the fixes' CPU tests, the 600- and 4,096-token queries, and the native 600-token `--det 1` reference structure |
| `tools/compare_structures.py` | CA RMSD between two structures |
| `slurm/validate.sbatch` | the image against the native port on one node: weights fetch, check (byte for byte on one GCD), 2 and 8 GCDs |
