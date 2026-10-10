# colabfold: ColabFold on Setonix, with GPU MSAs

ColabFold 1.6.3 (AlphaFold2-Multimer v3) on Setonix's MI250X GCDs. Two things run on the GPUs:
- **Predictions** run through the ColabFold optimisation kit's modes.
- **MSAs** come from `colabfold_search` on MMseqs2's GPU build (`ports/mmseqs2-hip`).

The image holds the native port's tree at `/opt/colabfold`, with the same versions (`environment/requirements.lock`, the native venv's
freeze) and scripts. The evidence is in `af3-setonix/COLABFOLD_PORT.md`.

| mode | what it is | speed against stock (one GCD) | largest complex |
|---|---|---|---|
| `off` | stock ColabFold 1.6.3 | 1x | 4,235 residues on one GCD |
| `exact` | stock maths, with the parameters kept on the GPU; byte-identical to `off` under XLA's deterministic flags | ~1.03x | as `off` |
| `fast` | the kit's Pallas kernels and larger sub-batches | ~1.5-2.0x | 4,620 residues on one GCD |
| `big` | `fast` on one GCD; `--n_gpu 2/4/8` shares one complex across GCDs | each pass 3.4-4.0x faster on 8 GCDs | 11,165 residues on a node |

## The image

`ghcr.io/gbouras13/setonix-ports/colabfold`, built by `.github/workflows/colabfold-image.yml`.
- **No weights.** By default it reads Pawsey's AlphaFold2 parameters,
  `/scratch/references/colabfold_jun2026/database/alphafold2_multimer_v3`; `COLABFOLD_OPT_DATA_DIR` names another copy.
- **Databases** for `colabfold search` are Pawsey's ColabFold databases (`COLABFOLD_DB`), already GPU-padded.
- **Not yet validated on an MI250X.** The image's check is `slurm/validate.sbatch` (below).

```bash
sbatch --account=<project> slurm/pull_image.sbatch ghcr.io/gbouras13/setonix-ports/colabfold:main /scratch/<project>/$USER/singularity
```

## Run

```bash
#!/bin/bash
#SBATCH --account=<project>-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8,tmp:1000G      # the node's NVMe, for the databases (colabfold search)
#SBATCH --exclusive
#SBATCH --time=04:00:00
module load singularity/4.1.0-nohost
SIF=/scratch/<project>/$USER/singularity/colabfold_main.sif
singularity exec $SIF colabfold search queries.fasta msas/ --use-templates 1      # MSAs: GPU MMseqs2, databases staged to NVMe
singularity exec $SIF colabfold predict --mode fast msas/ out/ --num-recycle 3       # every a3m in msas/, one GCD; colabfold_batch options follow
```

- **`search`** takes ColabFold's FASTA: a complex is one record, with its chains joined by `:`.
  - It copies the three databases (522 GB) to the node's NVMe first, which takes ~4.5 min.
  - Set `STAGE=0` to read Lustre directly; that is only worth it for a handful of queries.
  - Options go to `colabfold_search`, e.g. `--af3-json` or `--pair-mode paired`.
- **`predict`** runs the port's runner, `scripts/cfs-run.sh`.
  - It sets the ROCm environment, and picks copy-only launch serialisation up to 2,310 residues and blocking above.
  - ColabFold reports a failed prediction as "Could not predict" and exits 0. The kit reports it as `FAILED` and exits non-zero.
- **Variables** (`CFS_SERIALISE`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.949` for 11,165-11,550 residues on 8 GCDs, `COLABFOLD_OPT_JIT_ROOT`)
  go inside the job script. Setonix starts every job from a fresh login environment (`SBATCH_EXPORT=NONE`).
- **Other commands:** `colabfold check`, `colabfold mmseqs ...` (the GPU binary), `colabfold version`, `colabfold python`,
  `colabfold shell`.

## Validate the image

`slurm/validate.sbatch` takes one GPU node for about 30 minutes. It checks three things:
- `colabfold check` on all 8 GCDs;
- the native bitwise check's four predictions, byte for byte against the native outputs (job 50480325);
- MSAs for 246 phage proteins with templates, natively and through the image, compared file by file.

```bash
sbatch --account=<project>-gpu ports/colabfold/slurm/validate.sbatch SIF=<the .sif> OUT=<dir> NATIVE_W=<native tree> \
  REF_PRED=<native tree>/out/bitwise1.50480325 NATIVE_MMSEQS=<dir with the native bin/mmseqs, mmseqs-dbload2, msa_gpu.sbatch>
```

## Files

| path | what |
|---|---|
| `scripts/cfs-run.sh`, `scripts/cf-env.sh`, `scripts/cfs-recipes.sh`, `scripts/cfs_tokens.py` | the runner, the ROCm environment, the levers each mode drops on gfx90a, the size rule |
| `scripts/boot/sitecustomize.py`, `scripts/bin/nvidia-smi` | the gfx90a boot patch (kit modes) and the GPU census shim |
| `scripts/cf-search.sh` | `colabfold search`: `msa_gpu.sbatch` as a command |
| `scripts/setup-venv.sh`, `scripts/make-kitpin.sh`, `tools/repin163.py` | the native install, and the re-pinned tree for ColabFold 1.6.3 |
| `tools/cf_outputs.py` | result summaries and run-against-reference comparisons |
| `container/` | Dockerfile, entry point `colabfold`, self-check, architecture prune |
| `slurm/validate.sbatch` | the image against the native port |
