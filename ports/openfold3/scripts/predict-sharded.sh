#!/bin/bash
# predict-sharded.sh — ONE prediction row-sharded across P GCDs of one Setonix node.
#
#   AF3_SETONIX_WORKDIR=... bash scripts/predict-sharded.sh <query.json> <P> [runner.yml]
#
# P must be 2, 4 or 8 and no more than the GCDs the job holds.
#
# ---------------------------------------------------------------------------
# WHY NOT `python -m openfold3_opt pred --mode big --n_gpu P`
# ---------------------------------------------------------------------------
# That is the kit's own launcher and it drags in the whole openfold3_opt mode
# machinery -- activation gates, pin checks, NVIDIA cell tables -- none of which
# applies here. Worse, its launcher shells out to `nvidia-smi` (tp.py:
# `launch_cc`, `gpu_census`), so it cannot enumerate AMD GCDs at all.
#
# The adapter does not need any of it. Its entry hook says so plainly:
#
#   "With OF3TP_WORLD > 1 and OF3TP_RANK set it installs openfold3_opt.tp_rowpair
#    the moment `openfold3.projects.of3_all_atom.runner` has been imported; with
#    OF3TP_WORLD absent or 1 it installs NOTHING."
#
# So a sharded run is: the hook directory FIRST on PYTHONPATH, the rank
# environment set, and stock `run_openfold predict` in each rank. The hook
# patches OpenFold3's modules onto opt_core.mem.rowpair as the runner imports.
#
# Rendezvous uses ROWPAIR_STORE (a file:// store) rather than a TCP port --
# opt_core's own note: on one node "nothing listens on a port, so there is no
# port to race for".
# ---------------------------------------------------------------------------

set -uo pipefail

QUERY="${1:?usage: predict-sharded.sh <query.json> <P> [runner.yml]}"
P="${2:?usage: predict-sharded.sh <query.json> <P> [runner.yml]}"
W="${AF3_SETONIX_WORKDIR:?set AF3_SETONIX_WORKDIR}"
KIT="${KIT_DIR:-$W/kit}"
# The SHARDED runner, not the single-GCD one: under the tp line the driver owns
# the triangle statements and refuses a yaml that also enables the module kernels.
RUNNER="${3:-$KIT/configs/setonix-mi250x-sharded.yml}"

case "$P" in 2|4|8) ;; *) echo "P must be 2, 4 or 8 (got $P)" >&2; exit 2 ;; esac

export PATH="$W/py312/bin:$PATH"
# Authoritative: the weights staged by scripts/stage-weights.sh. Do NOT inherit
# an OPENFOLD_CACHE pointing at $MYSOFTWARE -- that is the default project, which
# may be unwritable (PermissionError on runner.yml).
if [ -d "$W/openfold3-cache" ]; then
    export OPENFOLD_CACHE="$W/openfold3-cache"
else
    export OPENFOLD_CACHE="${OPENFOLD_CACHE:-$W/openfold3-cache}"
fi
export TMPDIR="${TMPDIR:-$W/tmp}" XDG_CACHE_HOME="${XDG_CACHE_HOME:-$W/xdgcache}"
source "$KIT/scripts/setonix-env.sh" >/dev/null
export PYTHONUNBUFFERED=1

OPT="$W/of3_opt_src/opt"
HOOK="$OPT/openfold3_opt/tp_rowpair/hook"
[ -d "$HOOK" ] || { echo "hook not found: $HOOK" >&2; exit 1; }

# Production defaults (2026-09-28; the confidence head 2026-09-30; ring priority, trimul hipBLASLt, RCCL channels
# 2026-10-02; triattn tiles + DiT LayerNorm skip 2026-10-03; atom-pair cache + size-limited one-pass samples 2026-10-04), each
# overridable from the environment:
#   triangle attention  opt_core's fused flash_triattn on fp16 operands (ROWPAIR_TRIATT_OPERANDS=fp16 needs
#                       patch_triatt_fp16.py; without the patch the operands stay fp32 and the dispatch falls back to
#                       the torch statement by name). OF3T_FLASH_IP must be an AMD precision: the kernel's default
#                       'tf32' does not build on gfx90a. Measured 2.94x trunk at 2,310 res; fp16's structural drift
#                       vs fp32 is in THROUGHPUT.md. Set ROWPAIR_TRIATT_CORE=torch for the exact fp32 statement.
#   triangle mult.      fpf_v4 (Triton; compiles on gfx90a; trunk checksums identical to torch).
#   query blocks        ROWPAIR_QBLOCK_SPLIT_TAIL=1 (patch_qblock_tail.py): the last query block no longer absorbs
#                       the ragged tail, so the attention transient does not depend on N mod 256.
#   confidence head     ROWPAIR_TRIATT_AUTOCAST_OFF=1 (patch_triatt_autocast.py): OpenFold3 runs the confidence
#                       pairformer under autocast(float32), which the flash serve layer read as the compute dtype,
#                       so it refused those fp16 calls; with this they are served. 2.3x confidence head at 2,310 res;
#                       structures identical, confidence outputs within ~0.04 pLDDT (THROUGHPUT.md). Without the patch
#                       the variable is ignored. Set it to 0 for the fp32 confidence head.
#   ring comm priority  TORCH_NCCL_HIGH_PRIORITY=1 (2026-10-02): the trimul ring's RCCL stream at high priority, so the
#                       exchange overlaps the GEMMs instead of queueing behind them. Scheduling only: bitwise identical
#                       on the pinned hexamer (job 50251290); -10% per pairformer block at 6,930 res. Set 0 to compare.
#   trimul contraction  ROWPAIR_TRIMUL_HIPBLASLT=1 (2026-10-02, the user's choice; patch_trimul_hipblaslt.py): the ring's
#                       bf16 tile GEMM on hipBLASLt instead of the rocBLAS backend OpenFold3 forces on ROCm, 13.4 -> 3.4 ms
#                       per call. A further -9.7% per block at 6,930 res, none at 2,310/P=4. A rounding-order change
#                       (fp32 accumulation in both): hexamer structures move 0.14-1.86 A, inside a 5.6-12.5 A sample
#                       spread (THROUGHPUT.md). Peak +0.08 GiB per rank (hipBLASLt workspace). Without the patch the
#                       variable is ignored. Set it to 0 for upstream's rocBLAS numerics.
#   RCCL p2p channels   NCCL_NCHANNELS_PER_PEER=8 (2026-10-02, the user's choice): RCCL opens 2 channels per peer by
#                       default, which caps the trimul ring at 15 GB/s; 8 give 36 GB/s. Transport only: bitwise
#                       identical (job 50258341); -17% per block at 6,930 res, -19% trunk at 2,310/P=4 (50258339).
#                       Costs ~0.33 GiB of RCCL buffers per GCD, outside PyTorch's allocator: set it to 2 for runs
#                       at the memory ceiling.
#   triattn tiles       PTX_TRIATTN_CONFIG=128,32,2,4,1,1 (2026-10-03): BLOCK_M,BLOCK_N,ROWS,num_warps,num_stages,ORDER for
#                       opt_core's flash_triattn. gfx90a reports compute capability 9.0, so the kernel's own table gives
#                       it the H100 default (64x32, 2 rows, 4 warps, 3 stages). These tiles: 2.92x per call (sweep, job
#                       50274397), triangle attention 1.91 -> 0.82 s per pair block at 6,930 res, trunk -26% at 2,310
#                       (jobs 50280429, 50280428). Same 32-key tiles and MFMA instruction, so every sum keeps its order:
#                       bitwise identical (trunk, structures, confidences). Read at import, for every head dim. Set it to
#                       empty for the kernel's own table.
#   DiT pair LayerNorm  ROWPAIR_DIT_LN_SKIP=1 (2026-10-03; patch_dit_ln_skip.py): OpenFold3 0.5.0 normalises the conditioned
#                       pair on every denoising step, but its only reader is the roll-out's pair-bias cache, full after a
#                       sample's first step; the LayerNorm is skipped then. Bitwise identical (job 50324562); diffusion
#                       26.1 -> 24.6 s per sample at 2,310 res / P=4. Without the patch the variable is ignored. Set 0 to compare.
#   atom-pair cache     ROWPAIR_ATOM_PAIR_CACHE=1 (2026-10-04; patch_atom_pair_cache.py): the atom encoder's pair path, both atom
#                       transformers' pair LayerNorm and the blocks' bias projections are step-invariant within a roll-out;
#                       computed on its first step and reused. Bitwise identical (job 50350501); diffusion 122.6 -> 86.9 s
#                       (5 samples) at 2,310 res / P=4, peak unchanged there. Holds ~0.7 GB per GCD per 2,310 res of atoms for
#                       the roll-out: set 0 for runs at the memory ceiling. Without the patch the variable is ignored.
#   shard transposes    ROWPAIR_TRANSPOSE_PEERS=auto (2026-10-05, the user's choice; patch_transpose_peers.py): the pair block's two
#                       in-place shard transposes exchange with every peer at once instead of one peer per round, while the
#                       staging (2 x 7 blocks of rows x Rmax x 128 fp32 at P=8: 5.4 GiB at 6,930 res) fits in half the free
#                       memory agreed over ranks; else one per round, as before. Bitwise identical (job 50409167); 18-mer
#                       trunk -3.3%, transposes 0.243 -> 0.143 s per block, nothing at 2,310 / P=4. Each rank logs its choice
#                       ("[transpose_peers] ..."). Without the patch the variable is ignored. Set 1 to compare.
#   LayerNorm kernel    ROWPAIR_LN_KERNEL=1 (2026-10-08, the user's choice; patch_ln_kernel.py): fp32 LayerNorm launches of width 32..128
#                       with >= 32,768 rows run a Triton row kernel (fp32 two-pass statistics, 16 rows x 4 warps) instead of ATen's
#                       vectorized kernel: 5.6-11.7x per call; 18-mer runtime -4.4%, hexamer -6% (jobs 50480300, 50480409). A
#                       rounding-order change: 18-mer CA 0.3-2.0 A against an 11-20 A sample spread, ranking unchanged. Each rank logs
#                       its first served launch ("[ln_kernel] ..."). Without the patch the variable is ignored. Set 0 for ATen.
export ROWPAIR_TRIATT_CORE="${ROWPAIR_TRIATT_CORE:-flash_triattn}" ROWPAIR_TRIMUL_KERNELS="${ROWPAIR_TRIMUL_KERNELS:-fpf_v4}"
export OF3T_FLASH_IP="${OF3T_FLASH_IP:-ieee}" ROWPAIR_TRIATT_OPERANDS="${ROWPAIR_TRIATT_OPERANDS:-fp16}"
export ROWPAIR_QBLOCK_SPLIT_TAIL="${ROWPAIR_QBLOCK_SPLIT_TAIL:-1}"
export ROWPAIR_TRIATT_AUTOCAST_OFF="${ROWPAIR_TRIATT_AUTOCAST_OFF:-1}"
export TORCH_NCCL_HIGH_PRIORITY="${TORCH_NCCL_HIGH_PRIORITY:-1}"
export ROWPAIR_TRIMUL_HIPBLASLT="${ROWPAIR_TRIMUL_HIPBLASLT:-1}"
export NCCL_NCHANNELS_PER_PEER="${NCCL_NCHANNELS_PER_PEER:-8}"
export PTX_TRIATTN_CONFIG="${PTX_TRIATTN_CONFIG-128,32,2,4,1,1}"
export ROWPAIR_DIT_LN_SKIP="${ROWPAIR_DIT_LN_SKIP:-1}"
export ROWPAIR_ATOM_PAIR_CACHE="${ROWPAIR_ATOM_PAIR_CACHE:-1}"
export ROWPAIR_TRANSPOSE_PEERS="${ROWPAIR_TRANSPOSE_PEERS:-auto}"
export ROWPAIR_LN_KERNEL="${ROWPAIR_LN_KERNEL:-1}"

# THE REPLICATED-DRAW POLICY. Unset, the adapter defaults to "guard": every rank
# draws the MSA subsample with its OWN generator and the values are proven
# identical afterwards. That assumes the generators stay in lockstep -- which
# they do not here, because rank 0 follows a different code path (it host-gathers
# the MSA and broadcasts the features, consuming device RNG draws the other ranks
# never make). The result is a silent divergence caught mid-trunk:
#
#   [rowpair r0] checksum[cycle0.m] identical_across_ranks=False
#                sum=-3.93e+05 shape=(1, 1024, 1155, 64)
#
# then rank 1 leaves the group and rank 0 blocks forever on the next collective.
#
# "bcast" removes the assumption: rank 0 draws and broadcasts the selection, and
# the VALUE is proven replicated instead of the generator state. The kit's own CLI
# always sets this (msa.py: "the tp line always sets it -- bcast at det 0, guard
# at det 1"); driving the hook directly means setting it ourselves.
export ROWPAIR_DIFF_NOISE_SYNC="${ROWPAIR_DIFF_NOISE_SYNC:-bcast}"

# ---------------------------------------------------------------------------
# THE tp LINE'S CONSTANTS (openfold3_opt/tp_rowpair/env.py: LINE_CONSTANTS)
# ---------------------------------------------------------------------------
# These are placements and policies the line RUNS WITH -- "not levers, nothing
# selects them". The kit's CLI writes them into every rank via export_core_env().
# Driving the hook directly means writing them ourselves, and omitting them is
# not benign:
#
#   ROWPAIR_MSA_HOST unset -> each rank uses ITS OWN MSA features instead of
#   rank 0's, so the MSA representation differs across ranks and the trunk's own
#   check fires mid-run:
#       checksum[cycle0.m] identical_across_ranks=False  shape=(1,1024,1155,64)
#   The host-parked keys (msa, has_deletion, deletion_value, token_bonds) are
#   deliberately EXCLUDED from the cross-rank feature digest, so nothing catches
#   this at featurisation time -- the digest reports ranks_identical=True and the
#   divergence only surfaces inside the model.
# DETERMINISM. The replication guard is BIT-EXACT ("a replicated input that
# differs across ranks is a broken run, never tolerated" -- dist.allreduce_checksum
# fingerprints the integer view). Replicated tensors are recomputed independently
# on each rank, so without deterministic kernels the last bits differ and the
# guard fires. Measured on the trimer at P=2:
#
#   rank 1  sum=-3.9342452165e+05  maxabs=7.904934e+01
#   rank 0  sum=-3.9342454244e+05  maxabs=7.904934e+01
#   -> relative difference 5.3e-8, i.e. float32 noise; maxabs identical.
#      Same values, different rounding -- not different MSA data.
#
# The hook applies the recipe (CUBLAS_WORKSPACE_CONFIG=:4096:8 +
# torch.use_deterministic_algorithms) when this is set, "so a stock run and this
# line share it". The kit's CLI sets it; driving the hook directly does not.
export OF3_DETERMINISTIC="${OF3_DETERMINISTIC:-1}"

# ROW-PARTITION ALIGNMENT. Without this the layout splits N by P with no
# alignment, so a rank's rows can start off the block grid and the driver refuses
# mid-run -- ON ONE RANK ONLY:
#
#   rank 1: refused: iter_row_blocks: rank rows start at 578, not a multiple of
#           the chunk unit                       (N=1155, P=2 -> 1155/2 = 577.5)
#   rank 0: blocks forever on collective 51, _ALLGATHER_BASE [578, 1024, 64]
#
# The sequence numbers show it exactly: rank 0 reached 51, rank 1 only 50 -- it
# refused and left before issuing the gather rank 0 was waiting for. That is the
# "hang": not a deadlock, one rank walking out.
#
# BLOCK_UNIT is 16 (floor.py: "the finest row-block unit the chunk plan reduces
# to"). The kit's CLI computes a larger chunk per run and halves it until every
# rank owns at least one full block (tp.chunk_of_run); 16 is the floor of that
# search and always partitions cleanly, at some cost in block size.
# MUST EQUAL settings.memory.eval.chunk_size in the sharded runner yaml: the
# driver aligns row blocks to the engine's chunk grid. 16 is the floor of the
# kit's search (BLOCK_UNIT) but NOT sufficient -- the unit is the engine chunk,
# and a mismatch refuses on one rank only while the others wait.
# Since 2026-10-03 it is READ FROM the runner yaml (below, once the yaml is
# snapshotted into the run directory), so the two cannot disagree; the sharded
# yamls carry chunk_size 128 (was 256). An explicit ROWPAIR_CHUNK_ALIGN that
# differs from the yaml stops the launch by name.

export ROWPAIR_MSA_HOST="${ROWPAIR_MSA_HOST:-rank0}"          # rank 0's MSA is authoritative; selected rows broadcast per recycle
export ROWPAIR_TEMPL_ORDER="${ROWPAIR_TEMPL_ORDER:-zparked}"  # z shard parked while the template pair stack runs
export ROWPAIR_TRIATT_STAGE="${ROWPAIR_TRIATT_STAGE:-once}"   # bias operand staged once per gathered plane
export ROWPAIR_DIFF_BIAS="${ROWPAIR_DIFF_BIAS:-ln_proj}"      # DiT pair-bias producer (refuses by name on ROCm -> engine statement)
export ROWPAIR_VERBOSE="${ROWPAIR_VERBOSE:-1}"                # per-rank census lines
export ROWPAIR_PARK_ZINIT="${ROWPAIR_PARK_ZINIT:-recompute}"  # no z_init tensor; recompute per row block
export ROWPAIR_HOST_SLAB="${ROWPAIR_HOST_SLAB:-lease}"        # one leased pinned slab per rank
export ROWPAIR_TRANSPOSE_INPLACE="${ROWPAIR_TRANSPOSE_INPLACE:-1}"
export ROWPAIR_FREE_ZTRUNK="${ROWPAIR_FREE_ZTRUNK:-1}"        # release the trunk shard for the roll-out
export ROWPAIR_CONF_PARK_ZTRUNK="${ROWPAIR_CONF_PARK_ZTRUNK:-1}"
export ROWPAIR_TEMPL_PARK_Z="${ROWPAIR_TEMPL_PARK_Z:-1}"
export ROWPAIR_TEMPL_FEAT_DEVICE="${ROWPAIR_TEMPL_FEAT_DEVICE:-cuda}"

# ---------------------------------------------------------------------------
# THE tp LINE'S ROW-BLOCK SIZES (modes.py _TP_ENV, mapped through
# tp_rowpair/env.py ENV_MAP onto the core's ROWPAIR_* names).
#
# These are NOT optional tuning. ROWPAIR_TRIATT_QBLOCK defaults to 0 = OFF
# (model.py: env_int("ROWPAIR_TRIATT_QBLOCK", 0)). With it off, triatt._stock_call
# takes the UNBLOCKED branch -- triatt.py:1460 `stock(q, k, v, biases)` -- instead
# of attend_query_blocks at :1459, and OF3's naive _attention then materialises
# the whole logits plane in one allocation:
#
#     rows * H * N^2 * 4 bytes        (rows = chunk 256, H = 4 heads, fp32)
#       N=1155 trimer  ->   5.09 GiB  (fits -- the trimer ran WITHOUT this lever)
#       N=2310 hexamer ->  20.36 GiB  (OOM on a 64 GiB GCD: job 49607297)
#       N=3465 9-mer   ->  45.80 GiB
#       N=4620 12-mer  ->  81.42 GiB  (hopeless)
#
# Row sharding does NOT shrink this: the key dimension spans all N columns on
# every rank, so the plane is N^2 no matter how the rows are split. The qblock
# lever is what bounds it, to 2*rows*H*qblock*N (API.md) -- 9.02 GiB for the
# 12-mer at qblock 256, 4.51 GiB at 128.
export ROWPAIR_TRIATT_QBLOCK="${ROWPAIR_TRIATT_QBLOCK:-256}"
export ROWPAIR_TRIATT_ROWBLOCK="${ROWPAIR_TRIATT_ROWBLOCK:-256}"
export ROWPAIR_APB_QBLOCK="${ROWPAIR_APB_QBLOCK:-256}"
export ROWPAIR_APB_ROWBLOCK="${ROWPAIR_APB_ROWBLOCK:-256}"
export ROWPAIR_TRIMUL_SUB="${ROWPAIR_TRIMUL_SUB:-128}"

# The remainder of LINES[("big","tp")].env in modes.py.
export OF3_FAST_INIT="${OF3_FAST_INIT:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OPENFOLD3_OPT_FASTJSON="${OPENFOLD3_OPT_FASTJSON:-1}"
export OPENFOLD3_OPT_LOADER_WORKERS="${OPENFOLD3_OPT_LOADER_WORKERS:-1}"
export OPENFOLD3_OPT_HOSTFEAT="${OPENFOLD3_OPT_HOSTFEAT:-1}"
export OPENFOLD3_OPT_CKPT_MMAP="${OPENFOLD3_OPT_CKPT_MMAP:-1}"

# modes.py _TP_UNSET: the other add-ons' levers conflict with this line.
unset OF3_CUDA_GRAPHS OF3_GRAPHS_STRICT OF3T_KIT_LEVERS OF3T_TEMPL_DISTINCT \
      OF3T_PAIRCACHE OF3T_TRIMUL OF3T_TRIATT OF3T_APB OF3TP_TEMPL_FIX \
      OF3TP_TEMPL_INTERCHAIN

# Above ~1000 residues rocBLAS asks for gfx90a Tensile GEMMs that are not in the
# shipped library and segfaults asynchronously; blocking launches make it fall
# back cleanly. Measured equivalent in cost to full serialisation.
# Templates are dense and UNSHARDED: template_distogram (1,4,N,N,39) +
# template_unit_vector (1,4,N,N,3) = 672*N^2 bytes on EVERY rank, 13.4 GiB at
# N=4620 and 30.1 GiB at N=6930. Row sharding does not touch them. Set false for
# an MSA-only run, which is what the reachable-N ladder measures.
export USE_TEMPLATES="${USE_TEMPLATES:-true}"

# With a query_msa.json from `run_openfold align-msa-server`, every chain already
# carries main_msa_file_paths. But --use-msa-server defaults TRUE and re-aligns
# anyway: measured on job 49624425, all 8 ranks each submitted to the ColabFold
# server and created their own msa-* dir despite valid precomputed paths. Set
# false alongside a precomputed query so the supplied alignments are used.
# (Cheap either way -- a homomer submits ONE sequence and takes ~2 s -- so this
# is about reproducibility and not hammering the server 8x, not about speed.)
export USE_MSA_SERVER="${USE_MSA_SERVER:-true}"

# 0 = asynchronous launches (1.17x at 2,310 res, trunk checksums identical). Clean -- zero crash lines -- at every
# size run since: 1,155 to 15,400 res, including the 36-mer (13,860) fit. 1 restores the old serialised launches.
export HIP_LAUNCH_BLOCKING="${HIP_LAUNCH_BLOCKING:-0}"

OUT="${OUT:-$W/out/sharded-${SLURM_JOB_ID:-local}-P${P}}"
LOGS="$OUT/ranks"
mkdir -p "$LOGS"

# THE RUNNER YAML, snapshotted into the run directory: the ranks read this copy, so an edit of the kit's yaml
# while the job runs cannot reach them (each rank reads the yaml at its own start). The row-partition alignment
# is taken from the copy's chunk_size (2026-10-03, the user's choice of the 128-row partition): -10.6% per pair
# block and -28% peak memory at 6,930 res / P=8, -15% trunk at 2,310 / P=4 (jobs 50309584, 50309585). It is a
# rounding-order change, not bitwise; a yaml with chunk_size 256 restores the old 256-row partition.
RUNNER_SRC="$RUNNER"
cp "$RUNNER_SRC" "$OUT/runner.yml" || { echo "cannot copy $RUNNER_SRC into $OUT" >&2; exit 2; }
RUNNER="$OUT/runner.yml"
YAML_CHUNK=$(python - "$RUNNER" 2>/dev/null <<'PY'
import sys, yaml
vals = set()
def walk(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "chunk_size" and isinstance(v, int) and not isinstance(v, bool):
                vals.add(v)
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)
walk(yaml.safe_load(open(sys.argv[1])))
print(",".join(str(v) for v in sorted(vals)))
PY
)
case "$YAML_CHUNK" in
    ""|*,*) echo "runner yaml $RUNNER_SRC: expected exactly one integer chunk_size, found '$YAML_CHUNK'" >&2; exit 2 ;;
esac
if [ -n "${ROWPAIR_CHUNK_ALIGN:-}" ] && [ "$ROWPAIR_CHUNK_ALIGN" != "$YAML_CHUNK" ]; then
    echo "ROWPAIR_CHUNK_ALIGN=$ROWPAIR_CHUNK_ALIGN, but $RUNNER_SRC has chunk_size $YAML_CHUNK: the two must match" \
         "(a mismatch refuses on one rank while the others wait). Unset ROWPAIR_CHUNK_ALIGN, or use a yaml with that chunk_size." >&2
    exit 2
fi
export ROWPAIR_CHUNK_ALIGN="$YAML_CHUNK"
STORE="$W/tmp/rowpair-store-${SLURM_JOB_ID:-$$}-P${P}"
rm -f "$STORE"

res=$(python3 -c "
import json
d=json.load(open('$QUERY'))['queries']
print(sum(len(c['chain_ids'])*len(c['sequence']) for q in d.values() for c in q['chains']))")

# ONE-PASS SAMPLES, A SIZE-LIMITED DEFAULT (2026-10-04, the user's choice). The tp line runs the diffusion roll-out and the
# confidence heads one sample per pass, so device memory does not grow with the sample count. ROWPAIR_SAMPLE_CHUNK=all
# (patch_sample_chunk.py) runs all samples in one pass: diffusion -48% at 2,310 res / P=4 and -31% at 6,930 / P=8 (jobs
# 50350036, 50379567). It is a rounding-order change: CA 0.08-1.32 A against a 5.8-12.5 A sample spread. The confidence
# head then holds the pair representation of every sample at once. The memory ladder (job 50379567) fits to N^2/P = 8.2M
# (8,085 res / P=8) and runs out of memory at 10.7M (9,240 / P=8), so it is the default while N^2/P <= ROWPAIR_ONEPASS_MAX_N2P
# (6.5M, ~20% under the largest fit: up to ~7,200 res on 8 GCDs, ~5,100 on 4, ~3,600 on 2). It also needs the runner
# yaml's per_sample_atom_cutoff: null (the sharded yamls carry it). N is this banner's residue count, so ligand atoms are
# not counted; a query it cannot count stays at one sample per pass. ROWPAIR_SAMPLE_CHUNK=1 | all chooses either way.
ONEPASS_MAX_N2P="${ROWPAIR_ONEPASS_MAX_N2P:-6500000}"
if [ -z "${ROWPAIR_SAMPLE_CHUNK:-}" ]; then
    PSAC=$(python - "$RUNNER" 2>/dev/null <<'PY'
import sys, yaml
ev = (((yaml.safe_load(open(sys.argv[1])) or {}).get("model_update") or {}).get("custom") or {}).get("settings", {}).get("memory", {}).get("eval", {})
print("null" if "per_sample_atom_cutoff" in ev and ev["per_sample_atom_cutoff"] is None else "set")
PY
)
    if ! [[ "$res" =~ ^[0-9]+$ ]]; then
        SC_WORD="one sample per pass (the query's residues could not be counted)"
    elif [ "$PSAC" != "null" ]; then
        SC_WORD="one sample per pass (the yaml's per_sample_atom_cutoff is not null)"
    elif [ $(( res * res / P )) -le "$ONEPASS_MAX_N2P" ]; then
        export ROWPAIR_SAMPLE_CHUNK=all
        SC_WORD="all in one pass (N^2/P = $(( res * res / P )) <= $ONEPASS_MAX_N2P)"
    else
        SC_WORD="one sample per pass (N^2/P = $(( res * res / P )) > $ONEPASS_MAX_N2P)"
    fi
else
    SC_WORD="ROWPAIR_SAMPLE_CHUNK=$ROWPAIR_SAMPLE_CHUNK, set in the environment"
fi

echo "=============================================================="
echo "  sharded prediction: $(basename "$QUERY")  ${res} residues"
echo "  P=$P GCDs   node=$(hostname)   runner=$(basename "$RUNNER_SRC")   row partition: ${ROWPAIR_CHUNK_ALIGN}-row chunks"
echo "  samples: $SC_WORD"
echo "  out=$OUT"
echo "=============================================================="

# Pre-fault torch's shared libraries into the NODE PAGE CACHE with one bulk
# sequential read before any rank starts. torch/lib is ~13 GB, and eight ranks
# mmap-faulting it concurrently off Lustre is pathological: measured on job
# 49630487, a node whose page cache had just been dropped by the previous job's
# OOM kill sat 46 MINUTES in cl_sync_io_wait having burned 2 s of CPU, and never
# launched a rank. Lustre itself was healthy throughout (all OSTs active,
# metadata 0.007 s, sequential read 585 MB/s) -- it is the random 4K mmap fault
# pattern that is slow, not the filesystem. One sequential read costs ~10-25 s
# and the import then takes ~3 s instead of stalling.
# Locate torch/lib WITHOUT importing torch: an import maps torch's 13 GB of .so files
# off Lustre before they are warm, the access pattern this step exists to avoid.
TORCHLIB=$(python -c 'import importlib.util,os;print(os.path.join(os.path.dirname(importlib.util.find_spec("torch").origin),"lib"))' 2>/dev/null)
echo "[launch] page-cache warm starting $(date +%T): $TORCHLIB"
t0=$(date +%s)
[ -d "$TORCHLIB" ] && cat "$TORCHLIB"/*.so > /dev/null 2>&1 || true
echo "[launch] page-cache warm: $(( $(date +%s) - t0 ))s  ($TORCHLIB)"
# Triton's compiled library (triton/_C, ~520 MB) and the AMD code-object compiler (backends/amd/lib, ~163 MB) are
# loaded by every rank on its first kernel; they sit outside torch/lib, so the read above misses them, and on a slow
# Lustre (7.5 MB/s on 2026-10-01) eight ranks faulting them in at once stalled whole jobs inside `import triton`.
# One sequential read here instead (~2 s on a healthy /scratch). find_spec locates the package without importing it.
TRITONPKG=$(python -c 'import importlib.util,os;s=importlib.util.find_spec("triton");print(os.path.dirname(s.origin) if s else "")' 2>/dev/null)
t0=$(date +%s)
[ -n "$TRITONPKG" ] && cat "$TRITONPKG"/_C/*.so "$TRITONPKG"/backends/amd/lib/*.so > /dev/null 2>&1 || true
echo "[launch] page-cache warm (triton): $(( $(date +%s) - t0 ))s  ($TRITONPKG)"

# One cold torch import off Lustre is ~3 min and EVERY rank would pay it.
t0=$(date +%s); python -c "import torch, openfold3" >/dev/null 2>&1
echo "[launch] warm import: $(( $(date +%s) - t0 ))s"
echo "[launch] settings: ROWPAIR_TRIATT_CORE=$ROWPAIR_TRIATT_CORE ROWPAIR_TRIATT_OPERANDS=$ROWPAIR_TRIATT_OPERANDS" \
     "ROWPAIR_TRIMUL_KERNELS=$ROWPAIR_TRIMUL_KERNELS ROWPAIR_TRIATT_AUTOCAST_OFF=$ROWPAIR_TRIATT_AUTOCAST_OFF" \
     "TORCH_NCCL_HIGH_PRIORITY=$TORCH_NCCL_HIGH_PRIORITY ROWPAIR_TRIMUL_HIPBLASLT=$ROWPAIR_TRIMUL_HIPBLASLT" \
     "NCCL_NCHANNELS_PER_PEER=$NCCL_NCHANNELS_PER_PEER PTX_TRIATTN_CONFIG=${PTX_TRIATTN_CONFIG:-kernel-table} ROWPAIR_CHUNK_ALIGN=$ROWPAIR_CHUNK_ALIGN" \
     "ROWPAIR_DIT_LN_SKIP=$ROWPAIR_DIT_LN_SKIP ROWPAIR_ATOM_PAIR_CACHE=$ROWPAIR_ATOM_PAIR_CACHE ROWPAIR_SAMPLE_CHUNK=${ROWPAIR_SAMPLE_CHUNK:-1}" \
     "ROWPAIR_TRANSPOSE_PEERS=$ROWPAIR_TRANSPOSE_PEERS ROWPAIR_LN_KERNEL=$ROWPAIR_LN_KERNEL" \
     "HIP_LAUNCH_BLOCKING=$HIP_LAUNCH_BLOCKING"
TRIMUL_PY="$W/opt_core_src/opt_core/opt_core/mem/rowpair/trimul.py"
if [ "$ROWPAIR_TRIMUL_HIPBLASLT" = 1 ] && [ -f "$TRIMUL_PY" ] && ! grep -q "\[trimul_hipblaslt\]" "$TRIMUL_PY"; then
    echo "[launch] note: ROWPAIR_TRIMUL_HIPBLASLT=1 is ignored: patch_trimul_hipblaslt.py is not applied ($TRIMUL_PY)"
fi
PAIRSTACK_PY="$W/opt_core_src/opt_core/opt_core/mem/rowpair/pairstack.py"
if [ "$ROWPAIR_TRANSPOSE_PEERS" != 1 ] && [ -f "$PAIRSTACK_PY" ] && ! grep -q "\[transpose_peers\]" "$PAIRSTACK_PY"; then
    echo "[launch] note: ROWPAIR_TRANSPOSE_PEERS=$ROWPAIR_TRANSPOSE_PEERS is ignored: patch_transpose_peers.py is not applied ($PAIRSTACK_PY)"
fi
MODEL_PY="$W/of3_opt_src/opt/openfold3_opt/tp_rowpair/model.py"
if [ "$ROWPAIR_LN_KERNEL" = 1 ] && [ -f "$MODEL_PY" ] && ! grep -q "\[ln_kernel\]" "$MODEL_PY"; then
    echo "[launch] note: ROWPAIR_LN_KERNEL=1 is ignored: patch_ln_kernel.py is not applied ($MODEL_PY)"
fi

# PERSISTENT TRITON KERNEL CACHE (2026-10-02, the user's choice). Each rank's cache used to live in the job's own
# directory ($AF3_SETONIX_LOCAL), so the first run of every job compiled every Triton kernel: 42 s of trunk and 59 s of
# wall at 2,310 res (job 50258341). Triton 3.8's file cache is safe to share across processes and jobs: put() writes each
# file into a private temp dir and os.replace()s it into place, and a kernel whose files are not all present is a miss
# (compiled again), so an entry is complete or absent. It lives in $W/triton-cache on /scratch, still one directory per
# rank index. When the 21-day purge deletes a kernel's files, that kernel is a miss and is compiled again. (/software
# was the first choice, but pawsey1018's /software quota is full: 257 of 256 GB on 2026-10-02.) Keys hash the kernel
# source, compile options and target, so an entry from older code or another Triton is never used.
# ROWPAIR_TRITON_CACHE= (empty) restores the per-job cache. A probe write checks space quota as well as permissions.
TRITON_PERSIST="${ROWPAIR_TRITON_CACHE-$W/triton-cache}"
if [ -n "$TRITON_PERSIST" ] && ! { mkdir -p "$TRITON_PERSIST" 2>/dev/null && echo probe > "$TRITON_PERSIST/.probe.$$" 2>/dev/null \
                                  && rm -f "$TRITON_PERSIST/.probe.$$"; }; then
    echo "[launch] triton cache: cannot write to $TRITON_PERSIST -- per-job cache instead"
    TRITON_PERSIST=""
fi
if [ -n "$TRITON_PERSIST" ]; then
    echo "[launch] triton cache: persistent, $TRITON_PERSIST/triton-r<rank> ($(ls -d "$TRITON_PERSIST"/triton-r0/*/ 2>/dev/null | wc -l) kernels for rank 0)"
else
    echo "[launch] triton cache: per job, $AF3_SETONIX_LOCAL/triton-r<rank> (the first run compiles every kernel)"
fi

pids=()
for r in $(seq 0 $((P-1))); do
    (
        export PYTHONPATH="$HOOK:$OPT${PYTHONPATH:+:$PYTHONPATH}"   # hook FIRST
        export OF3TP_WORLD="$P" OF3TP_RANK="$r"                     # activates the adapter
        # Each rank must see ONLY its own GCD. Otherwise PyTorch Lightning sees all
        # 8, puts its tensors on device 0, and the collective fails with
        #   "Attempt to perform collective on tensor not on device passed to
        #    init_process_group"
        # because opt_core bound the group to ROWPAIR_LOCAL_RANK instead. Narrowing
        # ROCR_VISIBLE_DEVICES makes each rank's GCD its own device 0, so Lightning
        # and the process group agree. (Do NOT also set HIP_VISIBLE_DEVICES -- the
        # two filters compose; see scripts/setonix-env.sh.)
        export ROCR_VISIBLE_DEVICES="$r"
        export ROWPAIR_WORLD="$P" ROWPAIR_RANK="$r" ROWPAIR_LOCAL_RANK=0
        export ROWPAIR_STORE="$STORE" ROWPAIR_TAG="tp"
        export ROWPAIR_NCCL_TIMEOUT_S="${ROWPAIR_NCCL_TIMEOUT_S:-3600}"
        # Each rank keeps its own JIT cache dir: concurrent writes to one Triton
        # cache from 8 ranks is a known corruption source. (Persistent since 2026-10-02, see above.)
        export TRITON_CACHE_DIR="${TRITON_PERSIST:-$AF3_SETONIX_LOCAL}/triton-r$r"
        export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
        mkdir -p "$TRITON_CACHE_DIR"
        # stdbuf -oL: line-buffer the rank log. Without it a rank that is killed
        # or times out loses everything still sitting in its stdout buffer -- which
        # is exactly the evidence a hang investigation needs.
        stdbuf -oL -eL run_openfold predict --query-json "$QUERY" --runner-yaml "$RUNNER" \
            --use-templates "$USE_TEMPLATES" \
            --use-msa-server "$USE_MSA_SERVER" \
            --output-dir "$OUT" > "$LOGS/rank$r.log" 2>&1
        echo "$?" > "$LOGS/rank$r.rc"
    ) &
    pids+=($!)
done

echo "[launch] $P ranks started; waiting"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done

echo
echo "--- rank exit codes ---"
for r in $(seq 0 $((P-1))); do
    printf "  rank %d: rc=%s\n" "$r" "$(cat "$LOGS/rank$r.rc" 2>/dev/null || echo '?')"
done
echo
echo "--- rank 0 tail ---"
tail -20 "$LOGS/rank0.log" 2>/dev/null
echo
echo "--- adapter activation (should name tp_rowpair) ---"
grep -hE "tp_rowpair|rowpair\] (SCHEDULE|RANKENV)" "$LOGS/rank0.log" 2>/dev/null | head -6
echo
if grep -q "Successful Queries:  1" "$LOGS/rank0.log" 2>/dev/null; then
    echo "RESULT: SHARDED PREDICTION OK (P=$P, ${res} residues)"
    python3 -c "
import json,glob
f=glob.glob('$OUT/*/*/timing.json'); print('  runtime_s:', round(json.load(open(f[0]))['runtime_s'],1) if f else 'NA')
g=sorted(glob.glob('$OUT/*/*/*sample_1_confidences_aggregated.json'))
d=json.load(open(g[0])) if g else {}
print('  avg_plddt:', round(d.get('avg_plddt',0),2), ' ptm:', round(d.get('ptm',0),3))" 2>/dev/null
    exit 0
fi
echo "RESULT: FAILED (P=$P) -- see $LOGS/"
exit 1
