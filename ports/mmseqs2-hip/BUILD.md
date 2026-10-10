# MMseqs2 with AMD GPU (HIP/ROCm) support on Pawsey Setonix: build recipe

This is the native route: you build the binary yourself. To use it without building, pull the image instead (README.md); it carries
the same binary, built the same way. The jobs named below are in `slurm/` next to this file.

## What this produces

`mmseqs`: a single MMseqs2 binary with its GPU search compiled for the MI250X (`gfx90a`) GCDs of Setonix's GPU
nodes. MMseqs2 gained a HIP/ROCm GPU backend upstream in June 2026 (commits `c7eee55` "ROCm/HIP GPU support" and
`e067bef` for libmarv). Upstream calls it a preview and ships it only as a Docker image (`master-rocm7`), not in the
releases or bioconda. This recipe builds it natively on Setonix.

- Source: `soedinglab/MMseqs2` master, commit `564f40d8857f4eca4e1dfe100c67c155b1933e70` (2026-09-30).
- Toolchain: ROCm 7.2.4 (Pawsey's unsupported module tree), GCC 14.3 (gcc-native/14, the PrgEnv default), CMake 3.31
  and Ninja (pip), Rust stable (rustup; MMseqs2 compiles the `block-aligner` crate).
- The binary carries an RPATH to `/software/setonix/rocm/rocm-7.2.4/lib`, so it runs on a GPU node without loading
  any module.

## Requirements

Everything comes from Setonix itself, except four downloads done once on a login node:
- the MMseqs2 git repository (~34 MB);
- Rust stable, minimal profile (~200 files);
- a Python venv with `cmake>=3.31,<3.32` and `ninja` (~1,500 files without ColabFold);
- optionally `colabfold==1.6.3` in the same venv, for `colabfold_search` (~13,000 more files).

The build tree is temporary. After the build only `install/bin/mmseqs` is needed.

Why not `/software`: Pawsey limits files on `/software` per user (the build tree is a few thousand files), so build
on `/scratch`, then copy the single binary wherever it should live.

## Step 1: downloads (login node)

```bash
ROOT=/scratch/<project>/$USER/mmseqs2-hip
mkdir -p $ROOT/{rust,logs,tests}
git clone https://github.com/soedinglab/MMseqs2.git $ROOT/src
git -C $ROOT/src checkout 564f40d8857f4eca4e1dfe100c67c155b1933e70      # or stay on master

# Rust, self-contained under $ROOT (does not edit shell profiles)
export RUSTUP_HOME=$ROOT/rust/rustup CARGO_HOME=$ROOT/rust/cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --no-modify-path --profile minimal --default-toolchain stable

# CMake >= 3.31 and Ninja (Setonix's cmake module is 3.30.5; upstream builds with 3.31)
python3 -m venv $ROOT/venv
$ROOT/venv/bin/pip install "cmake>=3.31,<3.32" ninja
$ROOT/venv/bin/pip install "colabfold==1.6.3"        # optional: colabfold_search
```

## Step 2: build (CPU node, debug partition)

`slurm/build.sbatch` is the whole build:

```bash
sbatch --account=<project> slurm/build.sbatch ROOT=$ROOT
```

It took 7 minutes on one debug node with 64 cores (job 50231366): 112 s to configure, 235 s to compile and link,
302 steps, first attempt. The result:
- `install/bin/mmseqs`, 32 MB; `mmseqs version` prints the commit hash.
- Linked against `libamdhip64.so.7`, `libhsa-runtime64.so.1` and `libamd_comgr.so.3` from
  `/software/setonix/rocm/rocm-7.2.4/lib` (RUNPATH), and the system's `libgomp`, `libstdc++`, `libz` and `libbz2`.
- `roc-obj-ls` lists the embedded GPU code objects as `hipv4-amdgcn-amd-amdhsa--gfx90a`.

The CMake call it makes:

```bash
module use /software/setonix/unsupported && module load rocm/7.2.4
export ROCM_PATH=/software/setonix/rocm/rocm-7.2.4 HIP_PATH=$ROCM_PATH HIP_PLATFORM=amd
export RUSTUP_HOME=$ROOT/rust/rustup CARGO_HOME=$ROOT/rust/cargo
export PATH=$ROOT/venv/bin:$CARGO_HOME/bin:$PATH CC=gcc CXX=g++
GCC_DIR=$(dirname "$(g++ -print-libgcc-file-name)")      # /usr/lib64/gcc/x86_64-suse-linux/14
cmake -S $ROOT/src -B $ROOT/build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$ROOT/install \
    -DHAVE_AVX2=1 -DENABLE_HIP=1 -DCMAKE_HIP_ARCHITECTURES=gfx90a \
    -DCMAKE_HIP_COMPILER=$ROCM_PATH/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$ROCM_PATH \
    -DCMAKE_HIP_FLAGS="--rocm-path=$ROCM_PATH --gcc-install-dir=$GCC_DIR" \
    -DCMAKE_PREFIX_PATH=$ROCM_PATH \
    -DOpenMP_C_FLAGS=-fopenmp -DOpenMP_C_LIB_NAMES=gomp \
    -DOpenMP_CXX_FLAGS=-fopenmp -DOpenMP_CXX_LIB_NAMES=gomp -DOpenMP_gomp_LIBRARY=$GCC_DIR/libgomp.so \
    -DCMAKE_INSTALL_RPATH=$ROCM_PATH/lib -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON
cmake --build $ROOT/build -j 64 && cmake --install $ROOT/build
```

Why each non-obvious flag is there:
- **`--gcc-install-dir`**: ROCm's clang compiles the GPU sources and links the executable. On its own it picks the
  newest GCC on the node (15), while the host code is compiled by GCC 14. This pins both to one libstdc++.
- **`OpenMP_*` with `gomp`**: keeps the whole binary on GCC's OpenMP runtime even though clang does the final link.
  Upstream's container recipe does the same.
- **`HAVE_AVX2=1`**: Setonix's EPYC 7763 and 7A53 CPUs are Zen 3, which has AVX2 but not AVX-512. It also switches off
  `-march=native`, so one binary serves both CPU and GPU nodes.
- **`CMAKE_HIP_ARCHITECTURES=gfx90a`** only: MI250X. One target keeps the build short. Upstream's image also builds
  gfx942 and generic targets.

## Step 3: check it

`slurm/test_gpu.sbatch BIN=$ROOT/install/bin/mmseqs QUERIES=tests/SAOMS1_phanotate.faa OUT=<dir>` is a 5-minute job on one GPU node
(job 50232402, gpu-highmem, 8 MI250X GCDs). It searches
the 246 proteins of Staphylococcus phage SAOMS1 (pharokka test data) with the GPU and with the CPU, and compares every
hit.

**pdb100** (`pdb100_230517`, GPU-padded, 65 MB):

| run | wall | hits (query, target) | against CPU ungapped |
|---|---|---|---|
| CPU, default k-mer prefilter, 64 threads | 1.6 s | 330 | the GPU finds all 330, plus 199 more |
| CPU, ungapped prefilter (`--prefilter-mode 1`, what the GPU runs), 64 threads | 17.7 s | 529 | reference |
| GPU, 1 GCD (`HIP_VISIBLE_DEVICES=0`) | 2.4 s (prefilter 0.23 s) | 529 | **identical: same pairs, E-values and bit scores** |
| GPU, 8 GCDs | 5.6 s | 529 | identical |
| GPU through `gpuserver`, first call / second call | 5.0 / 1.2 s | 529 | identical |

**UniRef30 2302 consensus sequences** (8.7 GB, the ColabFold prefilter target):

| run | wall | result |
|---|---|---|
| GPU, 8 GCDs, all 246 proteins | 32.0 s | |
| GPU, 8 GCDs, 10 proteins | 13.8 s | 32 hits, identical to the CPU's |
| CPU ungapped, 64 threads, the same 10 proteins | 82.5 s | |

The wall times include reading the database from Lustre and the CPU alignment stage. The GPU path reproduces the CPU
ungapped search exactly. It differs from MMseqs2's default CPU search only because that default is a k-mer
prefilter, a different and less sensitive algorithm.

## Step 4: clean up

Keep `install/bin/mmseqs` (copy it where it should live, with `scripts/mmseqs-dbload2` next to it) and delete `src`, `build`, `rust`
and the venv. `/scratch` is purged after 21 days without access, so keep a copy elsewhere, or rebuild, or use the image.

## Using it

On a GPU node no module is needed; the binary finds ROCm 7.2.4 through its RUNPATH. Without `--gpu` it is an ordinary
CPU MMseqs2 (AVX2), on any Setonix node.

**A GPU search.** The target must be in MMseqs2's padded GPU format: run `mmseqs makepaddedseqdb DB DB_pad` once.
Pawsey's `/scratch/references/colabfold_jun2026` databases are already padded: their `.dbtype` is `0x00080000`, the
GPU flag.

```bash
mmseqs createdb queries.fasta qdb
mmseqs search qdb targetDB_pad res tmp --gpu 1 --prefilter-mode 1
```

- **Which GPUs.** It uses every GCD it can see. Limit it with `HIP_VISIBLE_DEVICES`; Slurm already narrows
  `ROCR_VISIBLE_DEVICES` to the job's GCDs.
- **Many searches against one database.** Keep the database resident on the GCDs with `mmseqs gpuserver targetDB_pad
  --prefilter-mode 1 [--max-seqs N] &`, then add `--gpu-server 1` to each search. Use the same `HIP_VISIBLE_DEVICES`
  and `--max-seqs` as the server.

**ColabFold MSAs** (`pip install colabfold==1.6.3` gives `colabfold_search`). Pawsey's database keeps each database in
its own subdirectory, so name them relative to the database root:

```bash
DB=/scratch/references/colabfold_jun2026/database
colabfold_search --mmseqs /path/to/mmseqs --gpu 1 --threads 64 \
    --db1 colabfold_uniref30/uniref30_2302_db --db3 colabfold_envdb/colabfold_envdb_202108_db \
    --use-env 1 [--af3-json] \
    queries.fasta $DB out/
```

`slurm/msa_gpu.sbatch` wraps this for a whole GPU node. It stages the databases to NVMe, starts one `gpuserver` per
database, and runs `colabfold_search --gpu-server 1` through `mmseqs-dbload2` (see the Lustre section below):

```bash
sbatch --account=<project>-gpu slurm/msa_gpu.sbatch MMSEQS=/path/to/mmseqs queries.fasta out/ [--use-templates 1] [--af3-json]
```

For templates (`--use-templates 1`), `--db2` must be a bare name: `colabfold_search` writes `out/<--db2>` and
`out/<name>_<--db2>.m8`. So put all three databases in one flat directory of symlinks (or copies) and pass bare names
(`uniref30_2302_db`, `colabfold_envdb_202108_db`, `pdb100_230517`). `msa_gpu.sbatch` does this.

`msa_gpu.sbatch` ran end to end on the `gpu` partition (job 50421759). It took 752 s for 8 chains of 385-2,285
residues with templates: 278 s staging and 471 s of search.

A complex is one FASTA record with its chains joined by `:`. For complexes, `colabfold_search` also writes paired
MSAs.

**Into the structure predictors:**
- **OpenFold3:** use each query's `out/<name>.a3m` as the chain's `colabfold_main.a3m` (OpenFold3's ColabFold name,
  capped at 16,384 rows). List it in `main_msa_file_paths` and run with `--use-msa-server false`. Paired MSAs for
  heteromers go in `colabfold_paired.a3m` per chain (`paired_msa_file_paths`); splitting the combined complex a3m into
  those files is not automated here yet.
- **AlphaFold3-style inputs** (AF3, OpenDDE): `--af3-json` writes `<name>.json` with `unpairedMsa` and `pairedMsa`.
  AlphaFold3 re-pairs `pairedMsa` rows by the species in their headers, and ColabFold's rows are paired already.
  ColabFold 2's preview therefore rewrites the paired headers to one synthetic species per row
  (`rewrite_paired_descriptions`). Do the same, or check, before trusting heteromer pairing.
- **Protenix:** its own `scripts/colabfold_msa.py` runs `colabfold_search` and adds the pseudo taxonomy IDs Protenix
  pairs on. It accepts `--mmseqs_path /path/to/mmseqs --gpu 1 --gpu_server 1 --db1 ... --db3 ...`. Its split into
  `pairing.a3m` and `non_pairing.a3m` handles two-chain complexes.

## The Lustre bottleneck in ColabFold searches

With the GPU prefilter, most of a `colabfold_search` run is CPU time spent reading the databases.

For the 246 SAOMS1 proteins on one GPU node (job 50233013, both databases), the run took 1,445 s. Of that, the GPU
prefilter was ~380 s, and the four post-prefilter CPU stages took ~1,020 s:

| stage | UniRef30 | envDB |
|---|---|---|
| `expandaln` | 190 s | 144 s |
| `align` | 106 s | 119 s |
| `filterresult` | 55 s | 90 s |
| `result2msa` | 93 s | 116 s |

**Why:**
- **`colabfold_search` forces a load mode.** Pawsey's databases have no precomputed `.idx` index, so
  `colabfold_search` forces `--db-load-mode 0` (`colabfold/mmseqs/search.py:95-106`).
- **Every stage reads its whole database first.** In modes 0 and 3, each of these MMseqs2 modules first "touches"
  its whole target database: `touch = (par.preloadMode != PRELOAD_MODE_MMAP)` in `src/util/expandaln.cpp:107`,
  `src/alignment/Alignment.cpp:59` and `src/util/result2msa.cpp:46`. `Util::touchMemory` does this in one thread.
  That is 160-220 GB of `_seq`, `_aln` and `_seq_h` per database, read sequentially at ~0.9 GB/s from Lustre. The
  search iterations' internal `align` steps also touch the padded prefilter database: 8.7 GB for UniRef30, 38 GB for
  envDB.
- **The page cache cannot help.** A gpu-highmem node has 490 GB, and the two databases' files are ~224 GB and
  ~246 GB, so each run evicts the other database. A second run on the same node was no faster: its UniRef30
  `expandaln` took 351 s against 190 s cold.

**The fix, measured** on the same 246 proteins with the GPU prefilter through `gpuserver` (job 50235377, one
gpu-highmem node). All 246 MSAs were byte-identical in every variant.

| variant | wall | speed-up |
|---|---|---|
| Lustre, `colabfold_search` defaults (job 50233013) | 1,445 s | 1.00x |
| Lustre + `--db-load-mode 2` (`mmseqs-dbload2`) | 1,440 s | none: envDB gets fast, but UniRef30's `expandaln` takes 825 s of random faults |
| stage both databases to the node's NVMe: 522 GB, 16 readers | 270 s | once per job |
| NVMe, defaults | 815 s | 1.77x (1.33x including the staging) |
| **NVMe + `--db-load-mode 2`** | **691 s** | **2.09x (1.50x including the staging)** |

`msa_gpu.sbatch` does this by default (`STAGE=1`). It asks Slurm for the node's NVMe with
`--gres=gpu:8,tmp:1000G`; `/tmp` on a GPU node is a 3.5 TB NVMe, and a job gets 128 GiB of it unless it asks.
What remains after the fix is mostly the envDB GPU prefilter (~200 s) and UniRef30's `expandaln` (~130 s).

**GPU node against CPU node**, the same 246 proteins with the same binary (job 50233014; the CPU run uses the
default k-mer prefilter on one 128-core work node):

| | wall |
|---|---|
| CPU node | 3,981 s |
| GPU node, Lustre | 1,445 s (2.8x) |
| GPU node, NVMe + mode 2 | 691 s (5.8x) |

The MSAs agree:
- The GPU's are a little deeper (median 52 sequences against 38), because the ungapped prefilter is more sensitive.
- For the median protein the GPU MSA contains all of the CPU's hits; the median hit-set Jaccard is 0.95.
- The differences are in a few hypothetical proteins with few hits.

**Against the public MSA server** (job 50421759). The comparison used 8 chains whose server MSAs OpenFold3 had
cached in September: the HK97 capsid and 7 phage RBPs of 1,863-2,285 residues.

| chain | server hits | local GPU hits | hit-ID Jaccard | local has server's | server has local's |
|---|---|---|---|---|---|
| HK97 capsid | 6,035 | 6,123 | 0.77 | 88% | 87% |
| 7 RBPs | 1,907-2,268 | 1,661-2,312 | 0.50-0.62 | 60-79% | 68-76% |

- **Depth** is about the same.
- **Template hits agree:** where the server found templates, nearly all of its hits are among the local ones; two RBPs
  have none either way.
- **The lower overlap on the long RBPs is untested.** Likely causes: ColabFold's diversity filter (`--diff 3000`,
  qid bins) keeping different but equivalent members out of thousands of candidates; the GPU's more sensitive
  prefilter; and possibly newer databases on the server.
- **Whether the difference changes predictions has not been tested.**

## Caveats

- **Upstream status.** MMseqs2 calls ROCm support a preview, and this is master, not a release. Pin the commit you
  validated.
- **This binary only runs on Setonix-like systems.** It is built for `gfx90a` only (other AMD GPUs need
  `CMAKE_HIP_ARCHITECTURES` extended), and its RUNPATH points at Setonix's ROCm 7.2.4.
- **GPU results are not MMseqs2's default results.** The GPU runs the ungapped prefilter at maximum sensitivity. It
  reproduces `--prefilter-mode 1` on the CPU exactly, but finds more hits than the default k-mer prefilter, so MSAs
  can be deeper.
- **Database ordering.** ColabFold notes that the prebuilt GPU+CPU databases "can result in different results on
  MMseqs2-CPU due to database ordering" compared with the old databases.
