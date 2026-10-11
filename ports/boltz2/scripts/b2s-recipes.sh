#!/bin/bash
# b2s-recipes.sh -- the named lever drops each kit mode needs on gfx90a (MI250X, ROCm 7.2, torch 2.14.0+rocm7.2, triton 3.8.0),
# as measured in the smoke jobs of 2026-10-03/04 (BOLTZ2_PORT.md, "P1"). Source it, then pass the mode's word to b2s_launch:
#
#   source scripts/b2s-recipes.sh
#   B2S_KERNELS=off B2S_CARD_DROPS="$B2S_DROPS_EXACT" bash scripts/b2s-run.sh pred --mode exact ...
#
# Every entry leaves the row BY NAME through the kit's own card-drop table (modes.CARD_DROPS); the reason word is what was seen.
# B2S_KERNELS=off is required by every kit mode (no cuEquivariance on ROCm). HIP_LAUNCH_BLOCKING=1 (scripts/bz-env.sh default since
# 2026-10-04) is required above ~500 tokens by stock and kit alike; the recipes were confirmed under it at 576 and 1,056 tokens
# (smoke 8, job 50379463: every attempt rc 0; exact byte-identical to stock at both sizes). DEBUG_CLR_GRAPH_PACKET_CAPTURE=0
# (scripts/bz-env.sh default since 2026-10-09) is required by fast's sampler roll-out, which the fast recipes keep since that date.

# measured blockers, one word each
_B_OPM="fpf_opm:gfx90a_lds_64KB(needs_133KB)"                 # OutOfResources: H100 tile (TMA, 3 stages) needs 133,120 B of LDS
_B_PWA="fpf_pwa:gfx90a_lds_64KB(needs_98KB)"                  # OutOfResources: 98,304 B of LDS
_B_LN="exactln:needs_cuda.bindings_nvrtc"                       # not applied: compiles through cuda.bindings / NVRTC
_B_TRIMUL="fpf_trimul:native_row_cuda_payload"                  # the core's `native` row above 256 tokens: a sealed CUDA payload; in fast it refuses and the
                                                                #   next row's NVPTX asm crashes the worker; in big it runs, finite but 2.3x slower (smoke 4 G3)
_B_TRANS="fused_transition:v2_row_all_nan_gfx90a"               # its `v2` row (opt_core fpf_transition_v2) serves above 256 tokens and returns all-NaN
_B_ROLL="rollout:hipgraph_rollout_heap_corruption"              # fast: the captured roll-out (with dit_fused / align_jacobi64 inside) corrupts the heap,
_B_HOIST="dit_hoist:rides_rollout"                              #   unless DEBUG_CLR_GRAPH_PACKET_CAPTURE=0 (graph fix, job 50560965); since 2026-10-09 these
_B_JAC="align_jacobi64:rides_rollout"                           #   three are only the opt-out below

# exact: bitwise to stock --no_kernels (smoke 3 X2, smoke 4 X3/X4). dit_par / dit_mask are bitwise too under launch blocking (graph
# fix, job 50560965; smoke 2's 9.8 A ran without it), but they buy nothing here (HK97 sampler 19.8 s with them, 20.1 s without:
# graph fix X3, fastdef E1), so they stay off.
B2S_DROPS_EXACT="exact:align_aligncap:torch_pin_2.12;exact:exactln:needs_cuda.bindings_nvrtc;exact:exactln_resid:needs_cuda.bindings_nvrtc"
B2S_DROPS_EXACT+=";exact:dit_sba:nvrtc_cuda_c;exact:dit_smx:nvrtc_cuda_c;exact:dit_glue:nvrtc_cuda_c"
B2S_DROPS_EXACT+=";exact:dit_par:no_gain_gfx90a;exact:dit_mask:no_gain_gfx90a"
B2S_DROPS_EXACT+=";exact:msa_pwa_exact:pin_torch2.12;exact:msa_trans2_exact:pin_torch2.12"

# fast: completes, finite, within stock's band on 1BRS x2 (smoke 5 K2). Since 2026-10-09 (the user's decision) with its sampler
# roll-out (rollout + dit_hoist + align_jacobi64), which needs DEBUG_CLR_GRAPH_PACKET_CAPTURE=0 (bz-env.sh). HK97 trimer (fastdef F2,
# job 50579785): 264.5 s = 1.06x stock (0.97x without the roll-out), 1.25 A from stock.
B2S_DROPS_FAST="fast:$_B_OPM;fast:$_B_PWA;fast:$_B_LN;fast:$_B_TRIMUL;fast:$_B_TRANS"
# opt-out: fast WITHOUT its roll-out, as before 2026-10-09 (append to either fast recipe: "$B2S_DROPS_FASTP;$B2S_DROPS_NOROLL")
B2S_DROPS_NOROLL="fast:$_B_ROLL;fast:$_B_HOIST;fast:$_B_JAC"

# big, one GCD: finite only WITHOUT fused_transition, and the memory attachment (xl) cannot run without it (the kit: "no kernels-off
# memory line exists"; its exit gate then refuses for the missing xl_report, rc 3). Not a working memory mode yet: needs the
# transition provider to skip its v2 row on this card (a worker-side patch, P2).
B2S_DROPS_BIG1="big:$_B_OPM;big:$_B_LN;big:$_B_TRIMUL;big:$_B_TRANS"
B2S_DROPS_BIG1+=";big:xl_trans:needs_fused_transition;big:xl_cond:needs_fused_transition;big:xl_free:needs_fused_transition"
B2S_DROPS_BIG1+=";big:relpos_lazy:needs_fused_transition;big:expandable_segments:needs_fused_transition"

# big --n_gpu P: completes rc 0 (smoke 2 T1); also needs OF3T_FLASH_IP=ieee (the sharded trunk's flash triangle attention does not
# compile with its default tf32 input precision on gfx90a)
B2S_DROPS_BIGTP="big:$_B_OPM;big:$_B_LN"
export B2S_DROPS_EXACT B2S_DROPS_FAST B2S_DROPS_NOROLL B2S_DROPS_BIG1 B2S_DROPS_BIGTP

# ---- OPT-IN: the worker-side row patch (scripts/b2s_worker_boot.py), validated 2026-10-04 (valid 2 job 50387326, valid 3 job 50387597)
# B2S_WORKER_PATCHES=$B2S_PATCHES_GFX90A refuses two opt_core rows by name on gfx90a: the transition's `v2` at serve time (all-NaN; the
# provider steps aside to af3_fused) and every TriMul row but `v4`. With it, fused_transition and fpf_trimul stay on, and big keeps its
# whole memory line on one GCD. NOT a default: which kernel rows serve is a precision choice (the user's). Measured on the HK97 trimer
# (1,155 tokens, MSA): big 1.21x stock on one GCD, fast 1.12x, both rc 0, 0.4-1.7 A from stock against a 14-35 A seed spread.
# fast + patch WITH its roll-out (the recipe since 2026-10-09; graph fix K3/K4, fastdef F1 = K3): 1.22-1.24x, repeat byte-identical,
# 2.0 A from stock; still fits 2,310 tokens on one GCD (fastdef L6: alloc 44.4 GiB, +0.8 for the roll-out; reserved 61.0 unchanged).
B2S_PATCHES_GFX90A="transition_no_v2,trimul_v4_only"
B2S_DROPS_BIGP="big:$_B_OPM;big:$_B_LN"
B2S_DROPS_FASTP="fast:$_B_OPM;fast:$_B_PWA;fast:$_B_LN"
export B2S_PATCHES_GFX90A B2S_DROPS_BIGP B2S_DROPS_FASTP

# ---- the allocator under big (2026-10-05, capacity ladders). big's memory line applies the core's expandable_segments lever in the
# worker and in every --n_gpu rank WHATEVER the row says: a card drop "big:expandable_segments:..." only takes it off the parent's list,
# the worker then sets it through the runtime API, and the parent's gate exits rc 3 ("memory levers applied outside the row"; ladder1
# 50409262 B6/B7/B8, ladder8 50409327/50409328). So: no drop (the kit's allocator, rc 0), or, for an expandable-OFF run, BOTH
#   B2S_CARD_DROPS="...;big:expandable_segments:<why>"   (the parent's row)   and   B2S_WORKER_PATCHES=...,xl_no_expandable
# (scripts/b2s_worker_boot.py: big.row_settings composes the worker's line without the lever), so row and line agree.
# NOT the boot pin alloc_no_expandable: opt_core's `conf` rule refuses the lever, and that aborts the whole memory attachment
# ("REFUSED xl"; the worker never runs, rc 1; ladderc 50421726).
B2S_PATCH_XL_NO_EXPANDABLE="xl_no_expandable"
B2S_DROP_EXPANDABLE="big:expandable_segments:b2s_ab_expandable_off(xl_no_expandable)"
export B2S_PATCH_XL_NO_EXPANDABLE B2S_DROP_EXPANDABLE

# ---- the sharded line's speed words: DEFAULT since 2026-10-08 (the user's decision) -- scripts/b2s_launch.py sets them itself for
#      `pred --n_gpu P>1` (B2S_TP_SPEED=0 opts out); the variables below remain for reference and for other entries.
# (2026-10-07, speed8 job 50463489): BYTE-IDENTICAL output to the recipe without them, same peak
# memory, 1.79x on `big --n_gpu 8` at 5,390 tokens (trunk 570 -> 316 s). Both act in every rank:
#   triattn_tiles  (boot word) PTX_TRIATTN_CONFIG=128,32,2,4,1,1 for opt_core's flash triangle-attention kernel (the OpenFold3 retune;
#                  the kit strips PTX_TRIATTN_* from the worker environment, so the boot script sets it inside each rank)   -15.5%
#   RCCL words     NCCL_NCHANNELS_PER_PEER=8 TORCH_NCCL_HIGH_PRIORITY=1 (not stripped by the kit)                           -29.1%
# usage:  env $B2S_TP_SPEED_ENV B2S_WORKER_PATCHES=$B2S_TP_SPEED_PATCHES B2S_KERNELS=off OF3T_FLASH_IP=ieee \
#             B2S_CARD_DROPS="$B2S_DROPS_BIGTP" $RUN pred --mode big --n_gpu P --input X.yaml --out_dir OUT
B2S_TP_SPEED_PATCHES="triattn_tiles"
B2S_TP_SPEED_ENV="NCCL_NCHANNELS_PER_PEER=8 TORCH_NCCL_HIGH_PRIORITY=1"
export B2S_TP_SPEED_PATCHES B2S_TP_SPEED_ENV

# ---- DEFAULT since 2026-10-09 (the user's decision): big on ONE GCD on the kit's flash base -- scripts/b2s_launch.py applies the words
#      below itself for `pred --mode big` on one GCD (B2S_BIG_FLASH=0, or an explicit B2S_BIG_BASE=pairtrack, keeps the shipped base);
#      so the plain big command  B2S_KERNELS=off B2S_CARD_DROPS="$B2S_DROPS_BIGP" $RUN pred --mode big ...  runs it. Precision: its
#      triangle attention is MORE precise than stock's path (0.83-0.90x the error vs float64, job 50538802); it rounds differently.
# (2026-10-08, speed1b job 50480198). The shipped base (pairtrack) runs triangle attention
# on stock's chunked code here (its cuEquivariance core steps aside under --kernels off); the flash base serves opt_core's flash triangle-
# attention kernel, the sharded line's, with the tile word pinned to ROWS=1 (at ROWS>=2 the kernel's separate-mask scan does not compile on
# gfx90a / Triton 3.8, and every call falls back to stock). Measured: HK97 trimer (1,155 tok, upstream defaults) 107 s = 2.6x stock, 0.06 A
# from the shipped big run; HK97 k=6 (2,310 tok, 1 recycle / 10 steps) 174 s = 5.7x stock with LESS memory (39.5 vs 47.0 GiB alloc); fits
# 2,695, out of memory at 3,080 (= the shipped big's one-GCD ceiling). Not bitwise to the shipped big (flash kernel numerics, the sharded
# line's class).
# usage:  env $B2S_BIG_FLASH_ENV B2S_KERNELS=off B2S_WORKER_PATCHES=$B2S_PATCHES_BIG_FLASH B2S_CARD_DROPS="$B2S_DROPS_BIGP" \
#             $RUN pred --mode big --input X.yaml --out_dir OUT
B2S_BIG_FLASH_ENV="B2S_BIG_BASE=flash OF3T_FLASH_IP=ieee B2S_TRIATTN_CONFIG=128,32,1,4,1,1"
B2S_PATCHES_BIG_FLASH="trimul_v4_only,triattn_tiles"
export B2S_BIG_FLASH_ENV B2S_PATCHES_BIG_FLASH
