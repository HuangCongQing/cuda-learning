<!--
 * @Description: 
 * @Author: HCQ
 * @Company(School): UCAS
 * @Email: 1756260160@qq.com
 * @Date: 2022-02-02 20:08:32
 * @LastEditTime: 2022-05-02 10:14:51
 * @FilePath: /cuda-learning/README.md
-->
# cuda-learning
cuda学习入门

Note: https://www.yuque.com/huangzhongqing/hpc/pz921g

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
