# openfold3: OpenFold3 on Setonix, row-sharded over 2, 4 or 8 GCDs

OpenFold3 0.5.0 (`aqlaboratory/openfold-3`) on Setonix's MI250X GCDs, through the optimisation kit's row-sharded line (`tp_rowpair` on
`opt_core`), with the native port's gfx90a defaults. The evidence is in `af3-setonix/THROUGHPUT.md` and `RUNTIME.md`.

**Status: not yet validated on Setonix** (`slurm/validate.sbatch` compares this image's outputs with the native port's, byte for byte).

What the native port measured, today's defaults against the same port with every value-changing lever off ("stock settings"):

| measurement | result |
|---|---|
| HK97 18-mer, 6,930 residues, 8 GCDs (job 50501719) | 916 s against 9,446 s: **10.3x**. CA within 1.3-4.0 Å of stock settings, inside a 9.9-17.9 Å sample spread |
| HK97 hexamer, 2,310 residues, 4 GCDs (same job) | 177 s against 781 s: **4.4x** |
| 12 PDB entries released 2025-07 to 2026-09, 726-5,775 tokens (jobs 50560285-87) | **the same accuracy** against the deposited structures (TM-score, DockQ), 1.8-7.9x faster, 4.6x summed |
| largest complete prediction | 13,860 residues on 8 GCDs (the HK97 36-mer) |

## What makes it run, and run fast, on gfx90a

The image is the native port's tree, rebuilt at `/opt/openfold3` in the native layout, and its launcher is the native one, verbatim
(`scripts/predict-sharded.sh`; `scripts/setonix-env.sh`). The launcher's defaults, each documented in it with its job number:

1. **Sharding without the kit's CLI.** The kit's own launcher finds GPUs with `nvidia-smi`. The port starts one `run_openfold predict`
   per GCD with the kit's `tp_rowpair` hook first on `PYTHONPATH`, a file rendezvous, and the line's constants written out.
2. **Kernels for gfx90a.** opt_core's fused triangle attention on fp16 operands, with gfx90a tiles (`OF3T_FLASH_IP=ieee`,
   `PTX_TRIATTN_CONFIG`), and the confidence head served by it; `fpf_v4` for the triangle multiplication, its tile GEMM on hipBLASLt; a
   Triton row LayerNorm.
3. **Memory and communication.** The 128-row partition, the split query-block tail, all samples in one pass while N²/P ≤ 6.5M, the
   atom-pair cache, RCCL channels and stream priority, all peers in flight for the shard transposes.

The patches in `patches/` add these switches to the kit and two seed pins to OpenFold3 (`OF3_REF_POS_SEED`, `OF3_CONFORMER_SEED`,
off unless set). `port_of3_050.py` ports the kit's adapter, written for OpenFold3 0.4.1, to 0.5.0.

**The sources are the native port's, byte for byte.** The build clones OpenFold3 v0.5.0 (`c4771653`) and the kit (`f4f62fa6`), applies
the patches in the native order, and checks every file of the two kit trees and the three patched OpenFold3 files against
`environment/native-sources.sha256`, the native tree's own digests. Any difference stops the build. `openfold3 check` checks them again
at run time.

## Using the image

```bash
# 1. pull it (a compute job; Pawsey asks for pulls there)
sbatch --account=<project> slurm/pull_image.sbatch ghcr.io/gbouras13/setonix-ports/openfold3:main /scratch/<project>/$USER/singularity

module load singularity/4.1.0-nohost
# 2. the checkpoint, once (in a compute job): 2.29 GB from OpenFold3's public bucket, checked against the S3 object's checksum
singularity exec $SIF openfold3 fetch-weights /scratch/<project>/$USER/openfold3-weights

# 3. MSAs from the ColabFold server (in a compute job): writes DIR/query_msa.json, the query for predict
singularity exec $SIF openfold3 msa -i query.json -o msas

# 4. in a GPU job on a whole node (--exclusive; never --rocm)
singularity exec $SIF openfold3 check --weights $WEIGHTS                                   # GPUs, libraries, sources, checkpoint
singularity exec $SIF openfold3 predict --weights $WEIGHTS -i msas/query_msa.json -o out   # 2, 4 or 8 GCDs by size
singularity exec $SIF openfold3 predict --weights $WEIGHTS -i msas/query_msa.json -o out_stock --stock-settings
singularity exec $SIF openfold3 compare out_stock out                                       # bytes, CA RMSD, confidences
```

- Inputs are OpenFold3 query JSON. `-P auto` (the default) uses 2 GCDs up to 1,600 residues, 4 up to 3,100 and 8 above; `-P 2|4|8`
  chooses. Rank r runs on the node's GCD r, so run on a whole node.
- `--stock-settings` switches off every lever that changes values: fp32 PyTorch triangle attention, the confidence head off the fused
  kernel, rocBLAS for the trimul tile, one sample per pass, ATen LayerNorm and the 256-row partition. It is the port's precision
  reference, about 4x slower.
- Predictions are MSA-only by default, as every validated run was. `predict` refuses a protein chain without an MSA unless you add
  `--no-msa` (single sequence) or `--msa-server` (each rank asks the ColabFold server). `--templates` turns templates on; that is not
  validated in the image.
- Setonix starts every job from a fresh login environment (`SBATCH_EXPORT=NONE`). Set `OF3_WEIGHTS` (or use `--weights`) and any
  `ROWPAIR_*` / `OF3*` / `HIP_*` / `NCCL_*` setting inside the job script. The launcher documents each setting.
- For reproducible comparisons, pin the inputs: `OF3_REF_POS_SEED=42 OF3_CONFORMER_SEED=42`.
- Triton's compiled kernels persist in `WEIGHTS/.triton-cache` when the weights directory is writable, so only a job's first run
  compiles. `ROWPAIR_TRITON_CACHE=` gives a per-job cache instead.
- Two predictions at once in one job need different `-P`: the launcher names its rendezvous file by job and P.
- One GCD is not offered: the launcher shards over 2 or more. `openfold3 run_openfold ...` runs OpenFold3's own CLI in the port's
  environment, unvalidated.

## Files

| path | what |
|---|---|
| `container/Dockerfile` | the image: python-build-standalone 3.12.12, the lock, OpenFold3 v0.5.0 and the kit at `f4f62fa6` rebuilt with the patches and checked against the native digests; torch's ROCm kernels pruned to gfx90a |
| `container/openfold3`, `container/openfold3_check.py` | the entry point and its self-check |
| `environment/requirements.lock` | the 80 distributions the native port runs on, at its versions |
| `environment/native-sources.sha256` | the native tree's digests: every file of the two kit trees, and the three patched OpenFold3 files |
| `environment/cuda-prebuilt.txt` | the kit's prebuilt CUDA binaries, which the native tree never had; the image deletes them |
| `patches/` | the native port's patch scripts, in the order of the Dockerfile; five take the target path as an argument here |
| `scripts/`, `configs/` | the native launcher and environment script, and the two runner yamls (128-row defaults, 256-row stock settings), verbatim |
| `tools/` | the weights fetch and check, the output comparison, the query reader |
| `tests/test_entry.sh` | dry-run tests of the entry point's decisions (the image build runs them) |
| `slurm/validate.sbatch` | on one node: the weights fetch, the check, and four predictions byte for byte against the native port's |
