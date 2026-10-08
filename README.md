# setonix-ports

Biomolecular structure-prediction models ported to Pawsey's **Setonix** GPU nodes (AMD Instinct MI250X, gfx90a; ROCm 7.2.4). Each port
keeps the upstream model and adds what it needs to run correctly and fast on these GPUs: patches, settings, launchers and a wrapper
that picks validated defaults by input size. Images cannot be built on Setonix, so GitHub Actions builds each port's container and
publishes it on ghcr.io. Setonix then pulls it as a Singularity image (SIF).

| port | model | in this repository |
|---|---|---|
| [`af3jax`](ports/af3jax/) | AlphaFold 3 (JAX), one GCD or a whole node | wrapper, launchers, checks, image `ghcr.io/gbouras13/setonix-ports/af3jax` |
| `openfold3`, `protenix`, `boltz2`, `colabfold`, `opendde`, `mmseqs2-hip`, `proteinmpnn` | | not yet: ported and validated on Setonix, to follow `af3jax` here |

## Using an image on Setonix

1. Pull it as a SIF, in a compute job (Pawsey asks for pulls there, not on a login node):
   ```bash
   sbatch --account=<project> ports/af3jax/slurm/pull_image.sbatch ghcr.io/gbouras13/setonix-ports/af3jax:main
   ```
2. Check it on a GPU node: `singularity exec $SIF af3jax check`. It lists the GPUs and checks that every GPU library was loaded from
   the image.
3. Run it through the port's entry point, for example `singularity exec $SIF af3jax predict ...` (see the port's README).

- Load `singularity/4.1.0-nohost`. It binds `/scratch`, `/software` and `/data/references` into the container.
- Never add `--rocm`. It binds the host's ROCm 6.3.0 libraries, whose sonames match the image's 7.2.4 ones.
- No weights are inside any image. Bind your own.

## How the images are built

`.github/workflows/<port>-image.yml` builds on GitHub-hosted runners. A push to `main` that touches a port publishes `:main` and
`:sha-<commit>`, and a tag `<port>-vX.Y.Z` publishes `:X.Y.Z` and `:latest`. Pull requests build without publishing. The run's summary
gives the image digest and the pull command. `checks.yml` runs the fast checks (shell and Python syntax, ShellCheck, the Dockerfile build
checks, each port's dry-run tests) on every push.

GitHub publishes a new package as private. After a port's first build, open the package's settings on GitHub and change its visibility
to public. Setonix can then pull it without credentials.

## Licence

The contents of this repository are licensed under Apache 2.0 (`LICENSE`). The images contain third-party software under its own
licences, and no model parameters: see `NOTICE`.
