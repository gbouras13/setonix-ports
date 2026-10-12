# rfdiffusion3: RFdiffusion3 on Setonix, stock's designs at 1.37x — and one kernel that had to be fixed first

RFdiffusion3 (the `rfd3` model of `RosettaCommons/foundry` at `4010e3e2`) on Setonix's MI250X GCDs, through the optimisation kit's
`exact` mode. With the two gfx90a fixes below, `exact` produces **stock RFdiffusion3's designs** — 64 of 64 identical at production
scale, 40/40 on the nucleic-acid binder spec, 16/16 on the small-molecule one — and produces them faster. The evidence is in
`af3-setonix/RFDIFFUSION_PORT.md`.

| route | `diffusion_batch_size` | designs/hour, one GCD | one node | designs |
|---|---|---|---|---|
| `off` — stock `rfd3 design` | 8 | 427 | 3,414 | the reference |
| **`exact`** | 8 | **586** (1.37x) | **4,690** | **identical to stock** |
| `fast` | 8 | 619 (1.45x) | 4,955 | its own numerics |
| `off` — stock | 16 | 630 | 5,036 | |
| **`fast`** | **16** | **796** (1.86x) | **6,367** | its own numerics |

Measured at four roll-outs per spec key with eight processes on one node, so these are the per-GCD *and* node-saturated rates.
Upstream's own `compile_model=true` — which the kit calls "upstream's fastest documented environment" — is **227 designs/hour here, half
of plain stock**: a designed length drawn per design keeps handing inductor shapes it has not compiled.

**Self-consistency**, the field's own quality measure, through this repository's other two ports: 16 backbones → ProteinMPNN's 8
sequences each → ColabFold single-sequence → **16/16 self-consistent at scRMSD < 2 Å, median best-of-8 0.63 Å** (pLDDT 85–97).

## The two gfx90a fixes

`exact` is not stock's designs out of the box on this card, and stock itself is not even protein. Two things, neither of them an edit to
a kit file:

1. **apex's fused RMSNorm is wrong on a 64-lane wavefront.** Upstream binds `apex.normalization.fused_layer_norm.FusedRMSNorm` as *every*
   RMSNorm of the network, and the kernel says what it assumes (`csrc/layer_norm_cuda_kernel.cu:55`: `blockDim.x == warpSize`). It
   launches `(32, 4, 1)` blocks — four rows each — and rotates partial sums with `WARP_SHFL` at the **default** width, which is
   `warpSize` = 64 here, so the rows read each other's sums. Measured: **29 % relative error in fp32, and 0 of 16 designs protein-like**.
   `patches/apex-gfx90a-warp32.patch` passes the width explicitly (14 lines; a no-op on NVIDIA, where `warpSize` is already 32) and gives
   `torch.nn.RMSNorm`'s own error, 1.2–2.3e-7, and 152 of 152 designs protein-like. `rfdiffusion3 check` measures this against a float64
   reference, so the image can prove which kernel it has.
   A second patch, `patches/apex-rocm-build.patch`, only makes the pinned sources compile here (`cuda.h` has no hipify mapping; three
   `rsqrt` helpers are host functions calling the device-only `rsqrtf`). It changes no arithmetic.
2. **`RFD_BOOT=fzt_rows_off`** (`scripts/boot/sitecustomize.py`, applied in memory at interpreter start). The kit's fused SwiGLU
   transition keys its served row counts on compute capability, and gfx90a reports `(9, 0)` — H100's — so it fuses **every** row count,
   on the stated grounds that cuBLAS keeps one summation order there. This card's GEMMs are hipBLASLt's. With the lever on, `exact` is
   *not* stock's designs: one came out 20.2 Å away with a changed residue. The fix gives capability 9.0 an empty served-row range, so the
   lever takes the stock body **by name** — a route the kit defines for itself ("a stock route by design, not a fallback and not a partial
   activation"). It costs no measurable time (129 s against 131 s for 16 designs), and the kit's own `APPLIED` line then reads
   `fused_calls=0`. This lever is the *whole* of `exact`'s divergence on this card; the other three change nothing.

And one setting, `DEBUG_CLR_GRAPH_PACKET_CAPTURE=0` (`scripts/rfd-env.sh`): ROCm 7.2 corrupts the host heap replaying a captured graph.
With it, the kit's CUDA-graph sampler captures and replays here with **zero fallbacks**.

## The image

`ghcr.io/gbouras13/setonix-ports/rfdiffusion3`, built by `.github/workflows/rfdiffusion3-image.yml`.

- **Two interpreters.** This kit is a *file overlay* on the installed `rfd3` package, so stock and the kit modes can never share a
  site-packages: `venv-stock` is pristine and runs `--mode off` under `python -I`; `venv-kit` carries the ten-file overlay. They are
  built in one layer and then hardlinked together, so the image holds one copy of everything they share.
- **No checkpoint.** It is 2.7 GB and upstream distributes it, so the image mounts it exactly as the kit's own Dockerfile does.
  `rfdiffusion3 fetch /weights/rfd3` downloads and digest-checks it with the kit's own verb, on a node that has network.
- **apex built here**, from the kit's pinned commit with the two patches, against the image's own torch.
- **Pruned to gfx90a**, like the other ports: the torch wheel ships a complete ROCm user space, most of it kernels for other cards.
- **Not yet validated on an MI250X.** `slurm/validate.sbatch` is the check.

```bash
sbatch --account=<project> slurm/pull_image.sbatch ghcr.io/gbouras13/setonix-ports/rfdiffusion3:main /scratch/<project>/$USER/singularity
```

## Run

```bash
#!/bin/bash
#SBATCH --account=<project>-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
module load singularity/4.1.0-nohost
SIF=/scratch/<project>/$USER/singularity/rfdiffusion3_main.sif
W=/scratch/<project>/$USER/rfd3-weights          # holds rfd3/rfd3_latest.ckpt
export RFD_LOCAL=/tmp/rfd-$USER-$SLURM_JOB_ID && mkdir -p $RFD_LOCAL
C="singularity exec --bind $RFD_LOCAL:$RFD_LOCAL --bind $W:/weights $SIF rfdiffusion3"   # never --rocm

$C check                                          # versions, pins, both overlays, GPUs, and apex against float64
$C design --mode exact --det 1 \
    inputs=/opt/rfdiffusion3/foundry-4010e3e2/models/rfd3/docs/examples/protein_binder_design.json \
    out_dir=$PWD/out n_batches=1 seed=101
$C sanity $PWD/out                                # are they protein-like at all
```

The first run needs the checkpoint: `$C fetch /weights/rfd3` on a node with network (the `copy` partition, or a login node), once.

## Memory — the one limit to respect

Stock holds ~8 GiB of a GCD's 64 whatever you ask of it. The kit modes keep a roll-out cache and one captured graph per design length:

| route | `diffusion_batch_size` | holds | |
|---|---|---|---|
| `off` | any | 8 GiB | |
| `exact` | 8 | 40 GiB | fine |
| `exact` | 16 | 57 GiB | **runs out part way through a long run** |
| `fast` | 16 | 48 GiB | fine |
| `exact` | 32 | — | refused by name (exit 5): upstream's own rule sends the attention sparse |

So **`exact` at batch 8, `fast` up to 16**. It is not an allocator artefact: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` was tested
both ways and fails identically.

## One thing to know before comparing two runs

**RFdiffusion3 does not always repeat itself here, in stock.** Of upstream's own binder example the `pdl1` entry repeats exactly every
time, while `insulinr` does not always repeat even against itself — same backbone length, about one residue in ten different, and
occasionally a backbone several Ångström away. It is upstream's own (it happens running the identical command twice, in the pristine
interpreter, and `torch.use_deterministic_algorithms(True, warn_only=False)` does not raise), and it is why every claim above was read
against a stock-vs-stock floor measured in the same job. If you compare two runs and they differ, measure stock against itself first.

Upstream's `enzyme_design.json` example also refuses itself on every route ("Multiple ligand residues on chain B") — upstream's own input
validation on its own example, not the port.
