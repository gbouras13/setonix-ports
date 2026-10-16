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

- **Tags.**
  - `:main` is the newest build.
  - `:8907e667-1` will be the first release, ProteinMPNN commit `8907e667` plus port revision 1 (not published yet). Upstream's last
    tag, v1.0.1 (2022), is older than that commit, so the commit stands in for the upstream version.
  - `:8907e667` is the newest port revision of that commit.
  - A release is the validated image itself, with the same digest (`.github/workflows/release.yml`).
- **Weights included.** Upstream's repository carries both weight sets (71 MB); the kit's own `run.sh install --weights` checks their
  sha256 against `stock/PINS.json` at build time, for `vanilla` and `soluble` alike. No weights to fetch, no databases, no network.
- **Pruned to gfx90a.** The torch 2.14.0+rocm7.2 wheel ships a complete ROCm user space of its own; 6.2 GiB of it is GPU kernels for
  other cards (hipBLASLt 4.60 -> 0.21 GiB, rocBLAS 0.63 -> 0.31, aotriton 0.86 -> 0.10, MIOpen's databases 0.76 -> 0.06).
- **Validated on Setonix** (2026-10-11), image `sha-0bfef70`, digest `sha256:0e4f1eb5…` (job 50659780, every section passes):
  - `proteinmpnn check --design` on all 8 GCDs: every GPU library loaded from inside the image, torch's draw transform rocRAND's in
    float64, and 3HTN (429 residues) designed stock and `exact` inside the image, byte-identical.
  - The six upstream examples through native stock, native `exact`, image stock and image `exact`: every pairing byte-identical, 6 of
    6 designs each, single-GCD `exact` against native included.
  - `dp --mode exact` on 8 GCDs x 2 workers over 92 backbones: image and native byte-identical, 92 of 92. Designing runs at native
    speed (6.6 s for the first worker's share in both), but each image worker takes about 17 s longer to start, so this pass took
    40.7 s against 28.3 s. It is an A/A check at 92 backbones, not the 402 designs/s measured natively over 1,472.
  - At that size, 1,472 backbones x 8 designs (job 50668900), image and native are byte-identical too, on 8 GCDs x 2 and on one GCD.
- **Start-up.** Designing runs at native speed, but a run starts later from the SIF, and more so with more workers. At 1,472
  backbones x 8 designs (job 50668900, both arms warm) the difference is 5.7 s on one GCD (143.9 against 138.2 s), 9.7 s on 8 GCDs x 1
  (38.4 against 28.7) and 17.1 s on 8 x 2 (44.7 against 27.6). Singularity writes a SIF as a gzip squashfs, which each `singularity exec`
  mounts afresh and the kernel decompresses on one thread, so workers that start together queue for torch's libraries (job 50669530).
  - `slurm/pull_image.sbatch` re-compresses the SIF with lz4, which decompresses several times faster: on 8 x 2 at 1,472 backbones
    that SIF started 6.2 s behind native instead of 17.0 s, with byte-identical designs (job 50681452).
  - From the gzip SIF, `--per-gcd 1` was the faster whole node at that size (307 against 263 designs/s), because start-up grows with
    the number of workers; natively `--per-gcd 2` leads. From the lz4 SIF that margin should shrink to a second or two, inside the
    run-to-run spread (not measured). Even 263 is above stock's best node rate, 200.9.
  - For many short runs in one job, run them in one `singularity exec` (a second `dp` there started 12 s sooner), or unpack the image
    to the node's NVMe once and run from that: book it with `--gres=gpu:8,tmp:100G`, then `singularity build --sandbox /tmp/pms-sandbox $SIF`
    (about 13 s) and use `/tmp/pms-sandbox` in place of `$SIF`. Start-up then matches native (9.6 against 8.9 s for 16 backbones on 8 x 2).
  - These are warm figures. On a fresh node the order reverses, because native's 15 GB venv then comes cold from Lustre: the same
    1,472-backbone pass on 8 x 2 took 69.7 s from the SIF against 137.1 s natively (job 50681861; warm, 45.2 against 27.9). A job pays
    that once, so the SIF stays ahead for up to about four such passes per job.

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
  - `--mode exact` adds **`--hybrid_gemm 0`** for you, and says so. Fix 2 above: without it the kit refuses the mode by name and
    designs nothing. Its probe passes on a *single* backbone and fails from two, so the trap is that a one-input trial works and the
    real run does not. `--hybrid_gemm 1` asks for the lever anyway, and the kit will refuse it.
- **`dp`** spreads one pass over the GCDs (`scripts/pms_dp.py`). In `exact` it stays byte-identical to **one** stock process however many
  workers it uses: each worker computes every backbone's offset in the single-process RNG stream and designs only its own share of the
  kit's length-sorted batch plan. Natively `--per-gcd 2` was the fastest arrangement measured (402 designs/s against 339 at one worker
  per GCD); from a gzip SIF `--per-gcd 1` was faster at 1,472 backbones, and from the lz4 SIF the two are close (Start-up, above).
  In `off` each shard restarts the seed, so its designs are valid samples but not a single stock run's bytes.
- **Scratch.** `PMS_LOCAL` holds temporaries and the one compiled Triton kernel; bind it into the container as above, or the kernel is
  recompiled per job (about 2 s, job 50669530). `MODEL_OPT_JIT_ROOT` on `/scratch` keeps it between jobs.
- **Other commands:** `proteinmpnn check`, `proteinmpnn compare A B`, `proteinmpnn kit ...` (the kit's unpatched `run.sh`),
  `proteinmpnn version`, `proteinmpnn python`, `proteinmpnn shell`.

## Validate the image

`slurm/validate.sbatch` takes one GPU node for about 10 minutes:

- `proteinmpnn check --design` — versions, the kit's pin check for both weight sets, every GPU library loaded from inside the image, a
  matmul on each of the 8 GCDs, torch's uniform transform against both vendors', and upstream's 429-residue `3HTN` designed stock and
  `exact` **inside the image**, bytes compared. That input is past the graph-replay crash threshold, so it checks fix 3 too.
- the six upstream examples through four routes — native stock, native `exact`, image stock, image `exact` — all pairings compared byte
  for byte.
- a benchmark pass on all 8 GCDs, image against native, both warmed first: an A/A check at 92 backbones, where start-up is most of
  the wall clock, not a reproduction of the 402 designs/s measured over 1,472.

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
