# tests

The image carries one test input of its own: `inputs/smoke_1brs.json` (1BRS barstar + barnase, 199 tokens), which `esmfold2 check
--predict` folds through stock-fused and exact and compares byte for byte.

`slurm/validate.sbatch` compares the image with the native port on the 48 bench targets (`inputs/bench/ef2bench48.json`, 105-497
residues) and the 11 MSA inputs (`inputs/msa11.json`, ColabFold A3M files), read from the native tree, against the native outputs of
job valid6 (51248511, `out/valid6.51248511/`): every route through `scripts/ef2.sh`, seed 0, `--det 1`, library defaults.
