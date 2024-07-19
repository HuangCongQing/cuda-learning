#include <stdio.h>
#include <iostream>

template <typename Type>
__global__ void PrintGpuDataKernel(const Type* data, int start, int end) {
  for (int i = start; i < end; ++i) {
    if (std::is_integral<Type>::value) { // 是否是整数类型
      printf("| %.1f ", static_cast<float>(data[i]));
    } else {
      printf("| %.7f ", static_cast<float>(data[i]));
    }
  }
  printf("\n");
}
// PrintGpuDataKernel<Type><<<1, 1>>>(data, start, end);

template <typename Type>
void PrintGpu(const Type* data, int start, int end) {
  PrintGpuDataKernel<Type><<<1, 1>>>(data, start, end);
}
//  a host function call cannot be configured
template void PrintGpu<float>(const float* data, int start, int end);
template void PrintGpu<int>(const int* data, int start, int end);

// 两个向量加法kernel，grid和block均为一维
__global__ void add(float* x, float * y, float* z, int n)
{
    // 获取全局索引
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // 步长
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
    {
        // z[i] = x[i] + y[i];
        atomicAdd(&z[i], x[i] + y[i]); // 推荐使用
    }
}


int main(){
    int N = 1 << 20;
    int nBytes = N * sizeof(float);
    // step1: 申请host/device内存######################################
    // 1 申请host内存
    float *x, *y, *z;
    x = (float*)malloc(nBytes);
    y = (float*)malloc(nBytes);
    z = (float*)malloc(nBytes);
    // 初始化
    for(int i = 0;i<N;i++){
        x[i] = 0.0;
        y[i] = 1.0;

    }
    // 2 申请device内存
    float *d_x, *d_y, *d_z;
    cudaMalloc((void**)&d_x, nBytes);
    cudaMalloc((void**)&d_y, nBytes);
    cudaMalloc((void**)&d_z, nBytes);

    // step2: 将host数据拷贝到device!!!######################################
    cudaMemcpy((void*)d_x, (void*)x, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_y, (void*)y, nBytes, cudaMemcpyHostToDevice);


    // step3: 执行kernel<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<######################################
    // 定义kernel的执行配置
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    add<<< gridSize, blockSize >>>(d_x, d_y, d_z, N);
    
    // step4（optinal）：输出内容check######################################
    // 输出GPU的数值
    PrintGpu<float>(d_z, 0, 10);

    // 将device得到的结果拷贝到host
    // cudaMemcpy((void*)z, (void*)d_z, nBytes, cudaMemcpyDeviceToHost);
    // 检查执行结果
    // float maxError = 0.0;
    // for (int i = 0; i < N; i++)
    //     maxError = fmax(maxError, fabs(z[i] - 30.0));
    // std::cout << "最大误差: " << maxError << std::endl;

    // step5：释放内存######################################
    // 释放device内存
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    // 释放host内存
    free(x);
    free(y);
    free(z);
}
/*  
nvcc template.cu -o template  -L/usr/local/cuda/lib64 -lcudart -lcuda -lcublas
*/