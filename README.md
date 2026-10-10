# setonix-ports

Biomolecular structure-prediction models ported to Pawsey's **Setonix** GPU nodes (AMD Instinct MI250X, gfx90a; ROCm 7.2.4). Each port
keeps the upstream model and adds what it needs to run correctly and fast on these GPUs: patches, settings, launchers and a wrapper
that picks validated defaults by input size. Images cannot be built on Setonix, so GitHub Actions builds each port's container and
publishes it on ghcr.io. Setonix then pulls it as a Singularity image (SIF).

| port | model | in this repository |
|---|---|---|
| [`af3jax`](ports/af3jax/) | AlphaFold 3 (JAX), one GCD or a whole node | wrapper, launchers, checks, image `ghcr.io/gbouras13/setonix-ports/af3jax` (validated) |
| [`mmseqs2-hip`](ports/mmseqs2-hip/) | MMseqs2's GPU (HIP) build, for `colabfold_search` and any GPU search | image `ghcr.io/gbouras13/setonix-ports/mmseqs2-hip`, GPU test |
| [`colabfold`](ports/colabfold/) | ColabFold 1.6.3 with the kit's modes, plus `colabfold_search` on the GPU binary | image `ghcr.io/gbouras13/setonix-ports/colabfold`, validation job |
| [`proteinmpnn`](ports/proteinmpnn/) | ProteinMPNN inverse folding, byte-identical to stock and 11.6x faster per GCD | runner, boot patch, data-parallel driver, image `ghcr.io/gbouras13/setonix-ports/proteinmpnn`, validation job |
| `openfold3`, `protenix`, `boltz2`, `opendde` | | not yet: ported and validated on Setonix, to follow here |

## Using an image on Setonix

1. Pull it as a SIF, in a compute job (Pawsey asks for pulls there, not on a login node):
   ```bash
   sbatch --account=<project> slurm/pull_image.sbatch ghcr.io/gbouras13/setonix-ports/af3jax:main /scratch/<project>/$USER/singularity
   ```
2. Check it on a GPU node: `singularity exec $SIF af3jax check`. It lists the GPUs and checks that every GPU library was loaded from
   the image.
3. Run it through the port's entry point, for example `singularity exec $SIF af3jax predict ...` (see the port's README).

- Load `singularity/4.1.0-nohost`. It binds `/scratch`, `/software` and `/data/references` into the container.
- Never add `--rocm`. It binds the host's ROCm 6.3.0 libraries, whose sonames match the image's 7.2.4 ones.
- No weights are inside any image, except `proteinmpnn`, whose weights are part of the upstream repository (MIT). Bind your own.

## How the images are built

`.github/workflows/<port>-image.yml` builds on GitHub-hosted runners. A push to `main` that touches a port publishes `:main` and
`:sha-<commit>`. Pull requests build without publishing. The run's summary gives the image digest and the pull command. `checks.yml`
runs the fast checks (shell and Python syntax, ShellCheck, the Dockerfile build checks, each port's dry-run tests) on every push.

A release is not a rebuild. Once an image passes the port's checks on Setonix, `<port>-release.yml` (run from the Actions tab) gives that
same image, with the same digest, its version tags, and tags its source commit. A version is the upstream model's version plus a port
revision. `3.1.4-1` is the first release of this port on AlphaFold 3 3.1.4, and `3.1.4-2` would be a change to the port on the same
upstream release. Each release publishes three tags:
- `:<version>`, which never moves;
- `:<upstream version>`, the newest port revision of that release;
- `:latest`.

The images inherit this repository's public visibility, so Setonix pulls them without credentials (checked for `af3jax`). If a new
package shows as private, change its visibility in the package's settings on GitHub.

## Licence

The contents of this repository are licensed under Apache 2.0 (`LICENSE`). The images contain third-party software under its own
licences, and no model parameters: see `NOTICE`.
