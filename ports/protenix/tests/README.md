# tests

`test_launcher.sh` checks `scripts/ptx-pred.sh`'s decisions without a GPU or torch: the route, the command line and the environment
each route hands the model process. `.github/workflows/checks.yml` runs it on every push.

The image carries no test data of its own. `protenix check --predict` uses the kit's example
`kit/protenix_v2/opt/forward/flashpairformer/public_inputs/p199.json` (PDB 1BRS, barnase:barstar, 199 tokens, with precomputed
single-sequence MSAs), which is part of the kit's tree in the image.

`slurm/validate.sbatch` compares the image with the native port's own deterministic runs, read from the native tree:
- `out/vdet.50030425/{S1,K1}`: 1BRS at stock defaults, the stock CLI and the one-GCD big line (identical to each other);
- `out/spd2.50379609/n2048_p8_flashcomm`: a 2,048-token query on 8 GCDs (1 cycle, 10 steps, 5 samples) with the current defaults,
  whose repeat in the same job was byte-identical.
