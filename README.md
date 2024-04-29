<!--
 * @Description: 
 * @Author: HCQ
 * @Company(School): UCAS
 * @Email: 1756260160@qq.com
 * @Date: 2022-02-02 20:08:32
 * @LastEditTime: 2023-03-17 11:50:27
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
