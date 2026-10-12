# tests/

The image's own checks live in `container/rfdiffusion3_check.py` (`rfdiffusion3 check`), because they have to run inside the image.
This directory holds what a check needs and the build does not otherwise carry.

Nothing here yet: every input the checks use is already in the image — upstream's own example specs come inside the kit's source
archive (`foundry-4010e3e2/models/rfd3/docs/examples/`), and the only thing the image lacks is the checkpoint, which is mounted.
