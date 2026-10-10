# proteinmpnn: ProteinMPNN on Setonix, byte-identical and 11.6x faster

ProteinMPNN (`dauparas/ProteinMPNN` at `8907e667`) on Setonix's MI250X GCDs, through the optimisation kit's `exact` mode. Everything the
image produces is **byte-identical to stock ProteinMPNN** — same designs, same scores, same probabilities — it just produces them much
faster. The evidence is in `af3-setonix/PROTEINMPNN_PORT.md`.

| mode | what it is | designs/s, one GCD | designs/s, one node |
|---|---|---|---|
| `off` | stock `protein_mpnn_run.py` | 7.3 | 200.9 (64 processes) |
| `exact` | the kit's worker: one captured decode graph, batched backbones, a fused draw | **84.8** (11.6x) | **402.0** (2.0x stock's best node rate) |

Measured over 1,472 backbones x 8 designs = 11,776 designs, every one of them byte-identical to a single stock process — on 1, 8 and 16
workers alike. The kit's own read-once parser is 53x faster than upstream's and produces the same bytes, which is most of the gap at this
size (upstream's parser alone would add ~13 minutes).

The kit ships no `fast` tier for ProteinMPNN, so there is no accuracy trade-off to decide: `exact` is the whole port.

## The three gfx90a fixes

`exact` is not byte-identical out of the box on this card. Three things, none of them an edit to a kit file:

1. **The fused draw kernel computes cuRAND's uniform transform.** torch on ROCm draws through rocRAND — same Philox4x32-10, different
   conversion of random bits to a uniform. `scripts/pms_worker_boot.py` replaces the kernel in memory at start-up (`rocm_draw`, the
   default on a ROCm torch). The kit's own 512-draw probe cannot see this: the two transforms agree to about one part in a million, so
   the probe passes while the mode is quietly no longer exact.
2. **`--hybrid_gemm 0`.** The kit's own probe refuses that lever here (max |d| 2.6e-6) and is right to. Worth knowing why it was ever
   offered: gfx90a reports compute capability `(9, 0)`, which is H100's, so a capability-keyed table hands an MI250X NVIDIA's tuning.
3. **`DEBUG_CLR_GRAPH_PACKET_CAPTURE=0`** (`scripts/pms-env.sh`). ROCm 7.2 corrupts the host heap when a captured HIP graph is replayed a
   few hundred times, and `exact` replays one per residue. Without it, every input past ~100 residues dies in `CUDAGraph.replay()` with a
   SIGSEGV in ROCr's async-events thread or a glibc heap abort. See `af3-setonix/rocm-graph-bug/REPORT_AMD.md`; the same bug cost the
   Boltz-2 port a working optimisation until this flag was found.

## The image

`ghcr.io/gbouras13/setonix-ports/proteinmpnn`, built by `.github/workflows/proteinmpnn-image.yml`.

- **Weights included.** Upstream's repository carries both weight sets (71 MB); the kit's own `run.sh install --weights` checks their
  sha256 against `stock/PINS.json` at build time, for `vanilla` and `soluble` alike. No weights to fetch, no databases, no network.
- **Pruned to gfx90a.** The torch 2.14.0+rocm7.2 wheel ships a complete ROCm user space of its own; 6.2 GiB of it is GPU kernels for
  other cards (hipBLASLt 4.60 -> 0.21 GiB, rocBLAS 0.63 -> 0.31, aotriton 0.86 -> 0.10, MIOpen's databases 0.76 -> 0.06).
- **Not yet validated on an MI250X.** `slurm/validate.sbatch` is the check.

```bash
sbatch --account=<project> slurm/pull_image.sbatch ghcr.io/gbouras13/setonix-ports/proteinmpnn:main /scratch/<project>/$USER/singularity
```

## Run

```bash
#!/bin/bash
#SBATCH --account=<project>-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=01:00:00
module load singularity/4.1.0-nohost
SIF=/scratch/<project>/$USER/singularity/proteinmpnn_main.sif
export PMS_LOCAL=/tmp/pms-$USER-$SLURM_JOB_ID && mkdir -p $PMS_LOCAL
C="singularity exec --bind $PMS_LOCAL:$PMS_LOCAL $SIF proteinmpnn"       # never --rocm

$C design --mode exact --input backbones/ --out designs/ --num_seq_per_target 8 --sampling_temp 0.1 --seed 37   # one GCD
$C dp --mode exact --input backbones/ --out designs/ --gcds 0,1,2,3,4,5,6,7 --per-gcd 2 --num_seq_per_target 8  # the node
```

- **`design`** is the kit's own `run.sh design` with the boot patch injected. Every stock option works and means what it means upstream
  (`--seed`, `--num_seq_per_target`, `--sampling_temp`, `--pdb_path`, `--jsonl_path`, PSSM, residue dictionaries, ...). The kit refuses
  by name, exit 3, the few passes its worker does not re-state — `--ca_only`, `--score_only`, `--tied_positions_jsonl`,
  `--backbone_noise` above 0 — and `--mode off` runs those.
- **`dp`** spreads one pass over the GCDs (`scripts/pms_dp.py`). In `exact` it stays byte-identical to **one** stock process however many
  workers it uses: each worker computes every backbone's offset in the single-process RNG stream and designs only its own share of the
  kit's length-sorted batch plan. `--per-gcd 2` was the fastest arrangement measured (402 designs/s against 339 at one worker per GCD).
  In `off` each shard restarts the seed, so its designs are valid samples but not a single stock run's bytes.
- **Scratch.** `PMS_LOCAL` holds temporaries and the one compiled Triton kernel; bind it into the container as above, or the kernel is
  recompiled per job (~20 s). `MODEL_OPT_JIT_ROOT` on `/scratch` keeps it between jobs.
- **Other commands:** `proteinmpnn check`, `proteinmpnn compare A B`, `proteinmpnn kit ...` (the kit's unpatched `run.sh`),
  `proteinmpnn version`, `proteinmpnn python`, `proteinmpnn shell`.

## Validate the image

`slurm/validate.sbatch` takes one GPU node for under an hour:

- `proteinmpnn check --design` — versions, the kit's pin check for both weight sets, every GPU library loaded from inside the image, a
  matmul on each of the 8 GCDs, torch's uniform transform against both vendors', and upstream's 429-residue `3HTN` designed stock and
  `exact` **inside the image**, bytes compared. That input is past the graph-replay crash threshold, so it checks fix 3 too.
- the six upstream examples through four routes — native stock, native `exact`, image stock, image `exact` — all pairings compared byte
  for byte.
- a benchmark pass on all 8 GCDs, image against native, designs/s compared with the 402/s the native port measured.

```bash
sbatch --account=<project>-gpu ports/proteinmpnn/slurm/validate.sbatch SIF=<the .sif> OUT=<dir> NATIVE_W=<native tree>
```

## Files

| path | what |
|---|---|
| `scripts/pms-env.sh` | the per-job environment: venv, kit and checkout paths, the `nvidia-smi` shim, `DEBUG_CLR_GRAPH_PACKET_CAPTURE=0` |
| `scripts/pms_worker_boot.py` | the in-memory patches the kit worker starts under — `rocm_draw` and the bisection levers that found it |
| `scripts/pms_launch.py`, `scripts/pms-run.sh` | the kit's own entry point with the worker's command line routed through that boot |
| `scripts/pms_dp.py` | the data-parallel driver: one pass over many GCDs, still byte-identical to one stock process |
| `scripts/bin/nvidia-smi` | a sysfs-backed stand-in; the kit identifies the GPU with `nvidia-smi` alone and refuses `exact` without one |
| `tools/pms_compare.py`, `tools/pms_designdiff.py` | byte identity and per-design differences between two output directories |
| `tools/pms_drawcheck.py` | which uniform transform this torch build uses, and whether a fused kernel reproduces it |
| `container/` | Dockerfile, entry point `proteinmpnn`, self-check, architecture prune |
| `slurm/validate.sbatch` | the image against the native port |

`container/prune_rocm_arch.py` differs from the copies in `ports/af3jax` and `ports/colabfold`: it matches architecture tokens anywhere in
a file's path below the pruned directory, not only in its name, and knows the family targets `gfx110x`/`gfx115x`/`gfx120x`. torch's
`aotriton.images` is laid out as `amd-gfx90a/`, `amd-gfx942/`, ... with the architecture only in the directory, so the name-only rule kept
every card's kernels. The other two ports would gain a little from the same change; their images are already built and validated, so it is
left for whenever they are next rebuilt.
