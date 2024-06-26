<!--
 * @Description: 
 * @Author: HCQ
 * @Company(School): UCAS
 * @Email: 1756260160@qq.com
 * @Date: 2022-02-02 20:08:32
 * @LastEditTime: 2024-05-07 19:55:12
 * @FilePath: /cuda-learning/README.md
-->
# cuda-learning
cuda学习入门

Note: https://www.yuque.com/huangzhongqing/hpc/pz921g （暂未公开）

## 编译运行
```
mkdir build
cd build
cmake ..
make
```

```
nvcc hello-gpu.cu -o hello-gpu
# 调试
nvcc -g -G hello-gpu.cu -o hello-gpu
```


## PyTorch加入自定义Cuda算子demo
* docs: https://www.yuque.com/huangzhongqing/cuda/wqexr9
* code: [python_using_cpp_cuda](./python_using_cpp_cuda)
* ref: https://github.com/JeffWang987/Python_Using_Cpp_CUDA

```shell
# 编译cuda生成.so文件
python setup.py develop
# 测试运行
python ball_query_example.py

```


## CUDA相关库

* [cub](cuda_lib/cub)
* [thrust](cuda_lib/thrust)
* [cublas](cuda_lib/cublas)
* [cutlass](cuda_lib/cutlass)

### [cub](cuda_lib/cub)
* TODO

### [thrust](cuda_lib/thrust)
* TODO


### [cublas](cuda_lib/cublas)

下面是测试[demo](cuda_lib/cublas/test_gemm.cpp)
```shell
# 编译：
nvcc test_gemm.cpp -o test_gemm -L/usr/local/cuda/lib64 -lcudart -lcuda -lcublas
# 运行：
./test_gemm
```
链接：https://zhuanlan.zhihu.com/p/403247313



### [cutlass](cuda_lib/cutlass)
* TODO