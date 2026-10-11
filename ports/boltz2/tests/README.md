# tests

The image carries no test data of its own. `boltz2 check --predict` uses the kit's example `kit/boltz2/inputs/1BRS_x2_barnase_barstar.yaml`
(398 tokens, single sequence), which is part of the kit's tree in the image.

`slurm/validate.sbatch` compares the image with the native port on the HK97 capsid trimer with its ColabFold MSA (1,155 tokens), read
from the native tree (`queries/hk97/hk97_trimer_msa.yaml`), against the native outputs of job 50579785 (`out/fastdef.50579785/`): the
current defaults of every mode, seed 1, upstream settings.
