# What the image carries for its own checks

`proteinmpnn check --design` needs no stored outputs: it designs upstream's 429-residue `3HTN` twice, once through stock
`protein_mpnn_run.py` and once through the kit's `exact` worker, and compares the two byte for byte inside the image. That is the
port's whole claim, and it is self-contained — which matters once the native `/scratch` tree is gone.

The 429 residues are the point. ROCm 7.2's HIP graph packet capture corrupts the host heap after a few hundred replays, and the
`exact` worker replays one captured graph per residue, so an input this long also proves `DEBUG_CLR_GRAPH_PACKET_CAPTURE=0` is in
force. Upstream's two monomer examples (68 and 106 residues) pass either way and would not catch it.

Expected bytes from the **native** port go here once `slurm/validate.sbatch` has compared the two on Setonix, so that a later image
can be checked against the version that produced the numbers in `af3-setonix/PROTEINMPNN_PORT.md` rather than only against itself.
