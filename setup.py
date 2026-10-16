from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "flash_ulysses" / "csrc"

setup(
    ext_modules=[
        CUDAExtension(
            "flash_ulysses._C",
            sources=["flash_ulysses/csrc/bindings.cu",
                     "flash_ulysses/csrc/impl/flash_ulysses_kernel_hopper.cu"],
            include_dirs=[str(CSRC), str(ROOT / "thirdparty/ThunderKittens/include")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20"],
                "nvcc": [
                    "-O3", "-std=c++20", "-DKITTENS_SM90",
                    "-gencode=arch=compute_90a,code=sm_90a",
                    "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "--expt-relaxed-constexpr", "--extended-lambda",
                    "-lineinfo", "--ptxas-options=-v", "--fmad=false",
                ],
            },
            extra_link_args=["-lcuda", "-lrt"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
