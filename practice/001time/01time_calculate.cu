// https://www.yuque.com/huangzhongqing/cuda/qtvdpsqszqkwlqkz
// nvcc -std=c++11 01time_calculate.cu  -o myprogram

#include <stdio.h>
#include <iostream>
// #include <Eigen/Dense>
// using namespace Eigen;
using namespace std;
#include <chrono>   

constexpr int kIterations = 100;


// 两个向量加法kernel，grid和block均为一维
__global__ void add(float* x, float * y, float* z, int n)
{
    // 获取全局索引
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // 步长
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
    {
        z[i] = x[i] + y[i];
    }
}

int main()
{
    int N = 1 << 20;
    int nBytes = N * sizeof(float);

    // 申请托管内存（使用一个托管内存来共同管理host和device中的内存）
    float *x, *y, *z;
    cudaMallocManaged((void**)&x, nBytes);
    cudaMallocManaged((void**)&y, nBytes);
    cudaMallocManaged((void**)&z, nBytes);

    // 初始化数据
    for (int i = 0; i < N; ++i)
    {
        x[i] = 10.0;
        y[i] = 20.0;
    }

    // 定义kernel的执行配置
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    // 执行kernel

    float avg_time = 0.0;
    // forc循环次数
    for (int i = 0; i <= kIterations; ++i) {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        // model time location<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        add<<< gridSize, blockSize >>>(x, y, z, N);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "kernel:"
                << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                        begin)
                        .count()
                << "us" << std::endl;
        if (i > 0) {
        avg_time +=
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
                .count();
        }
        
    }
    std::cout << "Average kernel time:"
                << avg_time / kIterations << std::endl;

    

    // 同步device 保证结果能正确访问
    cudaDeviceSynchronize();
    // 检查执行结果
    float maxError = 0.0;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(z[i] - 30.0));
    std::cout << "最大误差: " << maxError << std::endl;

    // 释放内存
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

    return 0;
}