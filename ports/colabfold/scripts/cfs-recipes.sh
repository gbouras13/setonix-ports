#!/bin/bash
# cfs-recipes.sh -- the levers each kit mode runs WITHOUT on gfx90a (MODEL_OPT_LEVERS_OFF, the kit's own ablation switch), one reason word
# each, as measured. Sourced by cfs-run.sh when the caller has not set MODEL_OPT_LEVERS_OFF.
#   TRIATTN_XLA   the provider's pre-compiled CUDA triangle-attention bridge (sm_90a / sm_80 cubins through an XLA-FFI launcher): no ROCm
#                 build exists; AF_PALLAS_ATTN's binding keeps the pair-biased sites on the provider's portable rows.
#   pallas:fpf_core, pallas:fpf_trimul, pallas:fpf_block, pallas:fpf_transition
#                 the provider's own row switch (the kit passes `pallas:<row>` to the core): the fpf_pallas family checks the device itself
#                 and has no gfx90a tile table (`fpf_pallas: no_tiles -- no-tiles:gfx90a`); on the A100 column (OPT_CORE_PALLAS_CC=8.0) fpf_core
#                 heads the MSA-column cells and the 500-1,280-token triangle cells, and its refusal escapes the walk there as a plain error
#                 ("Could not predict", valid 1 job 50436361). Switched off by name, the walk serves the next measured row (rowshared@r2,
#                 cd_triatt, cd_trimul), each run on gfx90a against the stock statement (tools/probe_rows.py).
CFS_FPF_OFF="pallas:fpf_core,pallas:fpf_trimul,pallas:fpf_block,pallas:fpf_transition"
CFS_LEVERS_OFF_EXACT=""
CFS_LEVERS_OFF_FAST="TRIATTN_XLA,$CFS_FPF_OFF"
CFS_LEVERS_OFF_BIG="TRIATTN_XLA,$CFS_FPF_OFF"
