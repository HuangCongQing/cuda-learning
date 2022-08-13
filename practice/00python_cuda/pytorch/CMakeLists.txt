cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
# modify to your own nvcc path, or delete it if ok
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(add2 LANGUAGES CXX CUDA)

# python路径
# 如果使用的是非系统目录下的 Python 可以通过指定 Python3_ROOT_DIR 改变查找路径
# set(Python3_ROOT_DIR "${CMAKE_SOURCE_DIR}/venv")

find_package(Python REQUIRED)
# find_package(Python3 COMPONENTS Interpreter Development)

find_package(CUDA REQUIRED)

execute_process(
    COMMAND
        ${Python_EXECUTABLE} -c
            "import torch.utils; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE DCMAKE_PREFIX_PATH)

set(CMAKE_PREFIX_PATH "${DCMAKE_PREFIX_PATH}")

find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

# modify to your own python path, or delete it if ok
# include_directories(/usr/include/python3.7) 
include_directories(/home/hcq/anaconda3/envs/pcdet/include/python3.8) # 修改================================
include_directories(../include)

# 编译路径============================================================================
set(SRCS add2_ops.cpp ../kernel/add2_kernel.cu)
add_library(add2 SHARED ${SRCS}) # add2.torch_launch_add2

target_link_libraries(add2 "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}")