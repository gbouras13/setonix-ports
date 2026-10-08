# Optional: a prebuilt AlphaFold 3 wheel

Drop `alphafold3_open-3.1.4-cp312-cp312-linux_x86_64.whl` here to have the image install it instead of compiling one. With the default
`WHEELS_FROM=auto`, a wheel found here must match `STOCK_WHEEL_SHA256` in `../Dockerfile`: the wheel the Setonix port validated on
2026-09-22 (natively at `af3jax-setonix/wheels/`). Wheels are not committed (`.gitignore`), so the GitHub Actions build always compiles
its own, the way upstream builds it. That wheel's bytes differ from the validated one, because the build takes the wwPDB chemical
component dictionary of its build day.
