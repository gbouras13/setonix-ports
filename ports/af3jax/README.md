# af3jax: AlphaFold 3 (JAX) on Setonix

AlphaFold 3 v3.1.4, from the `sokrypton/alphafold3` JAX fork that the af3_jax kit pins, on jax 0.10.2 with AMD's ROCm plugin and ROCm
7.2.4. This port adds:
- the fixes that make it run correctly on MI250X: two source patches, XLA settings and copy-only launch serialisation;
- speed-ups chosen by one command, `af3jax predict` (the wrapper `scripts/af3jax_predict.sh`), from the input's size:

| setup | GPUs | tokens | what it does | against the plain network |
|---|---|---|---|---|
| A | one GCD | up to 3,584 | flash triangle attention, diffusion speed-ups, hipBLASLt GEMMs up to 3,456 | 3.5× monomer, 4.5× 4-mer, 4.2× 6-mer |
| B | one GCD | 3,585 to 4,480 | flash plus memory savings, hipBLASLt GEMMs with the pair-logits fix | the plain network does not fit one GCD; hipBLASLt is 1.12× over Triton GEMMs (10-mer) |
| C | 8 GCDs | above 4,480, to about 9,300 | the model row-sharded over the node, plus flash | 1.5× for one cold run, 2.0× on a warm seed |

These are the release check A1's numbers (job 50480402, natively, on the official parameters, through the wrapper). All 15 runs
passed, and every chain of every structure folded. The monomer and 4-mer times are warm seeds at 10 recycles; the 6-mer is a cold seed
at 1 recycle. A token is roughly one residue. The wrapper's first log line gives its decision,
for example `bucket=3200 regime=A gemm=rocblas(auto) ... pairlogits_rows=off(off)`.

**A known AMD library fault.** Above about 18.6 million rows (4,314 tokens squared), hipBLASLt's default algorithm computes AlphaFold 3's
diffusion `pair_logits_projection` wrongly, with no error. The chains then misfold silently. The wrapper avoids this by computing that
projection in 1,024-row chunks (`PERF_PAIRLOGITS_ROWS`) for any one-GCD run above 3,968 tokens on hipBLASLt GEMMs. The values are
unchanged. Leave it on: the wrapper warns if it is turned off.

## The image

`ghcr.io/gbouras13/setonix-ports/af3jax`, built by `.github/workflows/af3jax-image.yml`.
- **Tags.**
  - `:main` is the newest build.
  - `:3.1.4-1` is the first release, AlphaFold 3 3.1.4 plus port revision 1, once its Setonix check passes.
  - `:3.1.4` is the newest port revision of AlphaFold 3 3.1.4.
  - A release is the validated image itself, with the same digest (`.github/workflows/af3jax-release.yml`).
- It holds an af3_jax tree at `/opt/af3jax`, with the same wrapper and launchers a native install runs.
- **No weights.** For the official AlphaFold 3 parameters, pass `--model_dir DIR` (a directory holding `af3.bin.zst`). For the
  OpenFold3-preview2 weights converted to AF3 format, bind their directory at `/opt/af3jax/weights/p2`.
- **Not yet validated on an MI250X.** The native stack is validated. The image's check is `slurm/a1audit.sbatch` in container mode,
  against the native A1 run (below).

```bash
sbatch --account=<project> slurm/pull_image.sbatch                      # -> $MYSCRATCH/singularity/af3jax_main.sif
```

## Run

```bash
#!/bin/bash
#SBATCH --account=<project>-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
module load singularity/4.1.0-nohost
SIF=$MYSCRATCH/singularity/af3jax_main.sif
export AF3JAX_CACHE=$MYSCRATCH/af3jax-cache          # compiled programs, reused by later runs of the same size
singularity exec $SIF af3jax predict --input_dir in/my_complex --output_dir out \
  --model_dir /scratch/<project>/$USER/af3_params -- --use_msa_server
```

- The input is AlphaFold 3's JSON, one input per folder. `-- FLAGS` pass through to `run_alphafold.py`.
- For eight inputs of at most 4,480 tokens, run eight of these at once with `--gcd 0` to `--gcd 7`.
- Use `af3jax predict --help` for every option, and `--dry_run` to see the plan without running it.
- Other commands: `af3jax check`, `af3jax version`, `af3jax python` and `af3jax shell`.

## Validate the image (container A1)

The 15-run release check runs every run through the image and compares each one with the native A1's same run (same input, same seeds).
It takes one `gpu-dev` node for about 2.5 h, and needs the HK97 test inputs (not distributed) and your own parameters:

```bash
A1_DATA=<native tree with inputs_ceiling/, inputs_perf/, ref/> AF3_PARAMS=<dir with af3.bin.zst> \
AF3JAX_SIF=$MYSCRATCH/singularity/af3jax_main.sif A1_REF=<native tree>/out_A1.50480402 \
sbatch --account=<project>-gpu slurm/a1audit.sbatch
```

Without `AF3JAX_SIF` it audits a native tree (`AF3JAX_W`) instead.

## Files

| path | what |
|---|---|
| `scripts/af3jax_predict.sh` | the wrapper (v13): setups, GEMM library, pair-logits fix, cache, serialisation |
| `scripts/perf_launch.py`, `scripts/rowpair_launch.py` | one-GCD speed levers; the 8-GCD row-sharded launcher |
| `scripts/patch_rocm_capability.py`, `scripts/patch_tokamax_rocm.py` | the two ROCm source patches (applied to private copies per run) |
| `scripts/perf_cmp.py`, `scripts/compare_structs.py` | same-seed comparison of runs: structures, confidences, timings |
| `scripts/g4_gemm.py`, `scripts/g5_autotune.py`, `scripts/pl_test.py`, `cases/` | the hipBLASLt investigation's tools and cases |
| `container/` | Dockerfile, entry point `af3jax`, self-check, architecture prune |
| `slurm/pull_image.sbatch`, `slurm/a1audit.sbatch` | pull the image on Setonix; the release check |
| `tests/test_wrapper.sh` | the wrapper's decision table, as a dry run (CI) |
