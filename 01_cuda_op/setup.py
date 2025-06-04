from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="add2",
    include_dirs=["include"],
    version='1.0',
    author='rtzhang',
    author_email='xxx@gmail.com',
    description='cppcuda example',
    ext_modules=[
        CUDAExtension(
            "add2",
            ["src/add2_ops.cpp", "src/add2_kernel.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)