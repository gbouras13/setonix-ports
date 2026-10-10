# mmseqs2-hip: MMseqs2 on Setonix's GPUs

MMseqs2 with its HIP/ROCm GPU backend, built for the MI250X GCDs (gfx90a). The backend is an upstream preview: it is on master, not in
a release. The image pins commit `564f40d8` (2026-09-30), the commit Setonix's native build validated, and compiles it the same way:
GCC 14, ROCm 7.2.4, gfx90a only, AVX2. On a GPU the search runs MMseqs2's ungapped prefilter, and its hits match the CPU's
`--prefilter-mode 1` exactly: the same pairs, E-values and bit scores (the native build, job 50232402). It is more sensitive than the
CPU's default k-mer prefilter.

The `colabfold` image carries this binary for `colabfold_search`.

There are two ways to get it:
- **Pull the image** (below). Nothing to build.
- **Build the binary yourself** on Setonix: `BUILD.md`. It covers four downloads on a login node, then `slurm/build.sbatch`, about 7
  minutes on one CPU node. The result is a single binary with an RPATH to Pawsey's ROCm 7.2.4 module, so it needs no container or
  module to run. `slurm/msa_gpu.sbatch` runs ColabFold MSAs with it, and `slurm/test_gpu.sbatch BIN=...` checks it.

## The image

`ghcr.io/gbouras13/setonix-ports/mmseqs2-hip`, built by `.github/workflows/mmseqs2-hip-image.yml`. It contains Ubuntu 24.04 with AMD's
ROCm 7.2.4 HIP runtime and the binary, and nothing else. Run it without `--rocm`.
- **Validated on Setonix** (2026-10-10), image `sha-49d7cbd`, digest `sha256:4a92d596…`: `slurm/test_gpu.sbatch` (job 50596117)
  passes. On 1 GCD, on 8 GCDs and through `gpuserver`, the GPU gives the CPU's ungapped hits exactly (529 of 529 pdb100 pairs, 32 of 32
  UniRef30), and so does the native build. The `colabfold` image carries this binary, and its MSAs are byte-identical to native.

```bash
sbatch --account=<project> slurm/pull_image.sbatch ghcr.io/gbouras13/setonix-ports/mmseqs2-hip:main /scratch/<project>/$USER/singularity
```

## Use

- **Target format.** A GPU search needs its target database in MMseqs2's padded format: run `mmseqs makepaddedseqdb DB DB_pad` once.
  Pawsey's ColabFold databases (`/scratch/references/colabfold_jun2026`) are already padded.
- **Which GCDs.** The search uses every GCD it sees. Narrow it with `HIP_VISIBLE_DEVICES`.

```bash
module load singularity/4.1.0-nohost
M="singularity exec $SIF mmseqs"
$M createdb queries.fasta qdb
$M search qdb /scratch/references/colabfold_jun2026/database/pdb100/pdb100_230517 res tmp --gpu 1 --prefilter-mode 1
```

- **Many searches against one database.** Keep the database resident on the GCDs: start `$M gpuserver TARGET_pad --prefilter-mode 1 &`,
  then add `--gpu-server 1` to each search.
- **`mmseqs-dbload2`** (in the image) runs the database-reading steps with `--db-load-mode 2`. It is the fix for `colabfold_search` on
  Lustre (BUILD.md, "The Lustre bottleneck").

## Check

`slurm/test_gpu.sbatch` repeats BUILD.md's correctness test on one GPU node, in about 10 minutes, through the image (`SIF=`) or on a
binary you built (`BIN=`). It runs the 246 SAOMS1 phage proteins against pdb100 and UniRef30 and checks:
- the GPU on 1 GCD, on 8 GCDs and through `gpuserver` gives the same hits as the CPU's ungapped search;
- with `NATIVE=`, it also gives the same hits as another binary.

```bash
sbatch --account=<project>-gpu ports/mmseqs2-hip/slurm/test_gpu.sbatch SIF=<the .sif> OUT=<dir> NATIVE=<native mmseqs>
sbatch --account=<project>-gpu ports/mmseqs2-hip/slurm/test_gpu.sbatch BIN=<your mmseqs> QUERIES=ports/mmseqs2-hip/tests/SAOMS1_phanotate.faa OUT=<dir>
```

MMseqs2 is GPLv3. Its source is upstream at the pinned commit; the builds are `BUILD.md` (native) and `container/Dockerfile`.
