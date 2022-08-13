/*
 * @Description: 提供一个PyTorch可以调用的接口。
 * @Author: https://github.com/godweiyang/NN-CUDA-Example/blob/master/pytorch/add2_ops.cpp
 * @Company(School): UCAS
 * @Email: 1756260160@qq.com
 * @Date: 2022-08-13 23:05:47
 * @LastEditTime: 2022-08-13 23:20:56
 * @FilePath: /cuda-learning/practice/00python_cuda/pytorch/add2_ops.cpp
 */
#include <torch/extension.h>
#include "add2.h"

// torch_launch_add2函数传入的是C++版本的torch tensor，
// 然后转换成C++指针数组，调用CUDA函数launch_add2来执行核函数。
void torch_launch_add2(torch::Tensor &c,
                       const torch::Tensor &a,
                       const torch::Tensor &b,
                       int n) {
    // practice/00python_cuda/include/add2.h
    launch_add2((float *)c.data_ptr(),
                (const float *)a.data_ptr(),
                (const float *)b.data_ptr(),
                n);
}

// 用pybind11来对torch_launch_add2函数进行封装，，然后用cmake编译就可以产生python可以调用的.so库。
// 但是我们这里不直接手动cmake编译，具体方法看下面的章节。
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_add2",
          &torch_launch_add2,
          "add2 kernel warpper");
}