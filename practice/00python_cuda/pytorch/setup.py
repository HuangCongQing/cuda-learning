from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 参考：/home/hcq/tensorrt/PCDet/setup.py
setup(
    name="add2",
    version="0.0.1+20220813",
    include_dirs=["include"],
    description='封装cuda算子用py调用学习',
    # ext_modules数组可以封装很多
    ext_modules=[
        CUDAExtension(
            "add2", # 命名
            ["pytorch/add2_ops.cpp", "kernel/add2_kernel.cu"], # 注意路径
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
# 编译运行 生成add2.cpython-38-x86_64-linux-gnu.so