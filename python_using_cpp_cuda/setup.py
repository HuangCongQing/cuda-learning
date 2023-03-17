from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

''' 
调用此算子示例：
from cpp_CUDA_code import pointnet_cuda
pointnet_cuda.ball_query_wrapper(xxx)
'''

if __name__ == '__main__':

    setup(
        name='example',
        version='0.0.0',
        description='Examples illustrating how to use c++ and CUDA in python.',
        install_requires=[
            'numpy',
            'torch>=1.1',
        ],
        author='Jeff Wang',
        author_email='wangxiaofeng2020@ia.ac.cn',
        license='Apache License 2.0',
        packages=find_packages(),
        cmdclass={
            'build_ext': BuildExtension,
        },
        ext_modules=[
            CUDAExtension(
                # 名字 ，调用`from cpp_CUDA_code import pointnet_cuda`
                name="cpp_CUDA_code.pointnet_cuda",
                sources=[
                    "cpp_CUDA_code/pointnet_api.cpp", #名字： ball_query_wrapper
                    "cpp_CUDA_code/ball_query.cpp",
                    "cpp_CUDA_code/ball_query_gpu.cu",
                ]   
            ),
        ],
    )
