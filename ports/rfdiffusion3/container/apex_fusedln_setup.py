"""apex 0.1 at the kit's pin, built for gfx90a: the python package plus the ONE CUDA extension upstream's RMSNorm uses.

Run from the apex checkout ($RFD_ROOT/apex, NVIDIA/apex at a1d527a8 — the commit stock/PINS.json pins):

    python ../scripts/apex_fusedln_setup.py bdist_wheel -d ../wheels/<variant>

Why not apex's own `setup.py --cpp_ext --cuda_ext`: at this commit apex's setup.py has no ROCm path — it reads the bare-metal
CUDA version from `$CUDA_HOME/bin/nvcc` at module level (setup.py:168), which does not exist here, and it then builds twelve
extensions (fused optimizers, MLP, SyncBN, multi-tensor kernels) that upstream RFdiffusion3 never imports. The one extension
`rfd3` needs is `fused_layer_norm_cuda`: upstream's import guard (rfd3/model/layers/layer_utils.py:13-23) binds
`apex.normalization.fused_layer_norm.FusedRMSNorm` as every RMSNorm of the network, and that class's forward imports
`fused_layer_norm_cuda` on first use. `import apex` pulls in `apex.normalization` and `apex.optimizers`, both of which import
their CUDA backends lazily (apex/__init__.py says so), so the other eleven extensions can be absent.

Sources, flags and the extension name are apex's own (setup.py:256-265), minus two nvcc-only flags hipcc does not take:
`-maxrregcount=50` (no hipcc equivalent) and `--use_fast_math` (RFD_APEX_FASTMATH=1 adds `-ffast-math`, hipcc's nearest).
torch's BuildExtension hipifies the two sources itself (IS_HIP_EXTENSION); PYTORCH_ROCM_ARCH picks the target.
"""
import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HIPCC = ["-O3"]
if os.environ.get("RFD_APEX_FASTMATH", "0") == "1":
    HIPCC.append("-ffast-math")

setup(
    name="apex",
    version="0.1",
    packages=find_packages(include=["apex", "apex.*"]),
    description="apex 0.1 (NVIDIA/apex a1d527a8) for gfx90a: python package + fused_layer_norm_cuda",
    ext_modules=[
        CUDAExtension(
            name="fused_layer_norm_cuda",
            sources=["csrc/layer_norm_cuda.cpp", "csrc/layer_norm_cuda_kernel.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": HIPCC},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
