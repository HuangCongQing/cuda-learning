#include <iostream>  
#include <cuda_runtime.h>  
#include <cuda_profiler_api.h>  
  
// CUDA核函数声明  
__global__ void doubleArray(float *a, int n) {  
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  
    if (idx < n) {  
        a[idx] *= 2.0f;  
    }  
}
  
int main() {  
    int n = 1024 * 1024; // 1M elements  
    size_t size = n * sizeof(float);  
  
    // 分配主机和设备内存  
    float *h_a = (float*)malloc(size);  
    float *d_a;  
    cudaMalloc(&d_a, size);  
  
    // 初始化数据  
    for (int i = 0; i < n; ++i) {  
        h_a[i] = float(i);  
    }  
  
    // 拷贝数据到设备  
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);  
    // ##########################################
    // 创建CUDA事件  
    cudaEvent_t start, stop;  
    cudaEventCreate(&start);  
    cudaEventCreate(&stop);  
    // 记录开始时间  
    cudaEventRecord(start, 0);  
    // ##########################################

    // 执行核函数  
    doubleArray<<<1024, 1024>>>(d_a, n);  
  
    // ##########################################
    // 等待核函数完成  
    cudaDeviceSynchronize();  
    // 记录结束时间  
    cudaEventRecord(stop, 0);  
    // 等待所有事件都完成，确保可以安全查询时间  
    cudaEventSynchronize(stop);  
    // 计算并打印时间  
    float elapsedTime;  
    cudaEventElapsedTime(&elapsedTime, start, stop);  
    std::cout << "Elapsed time: " << elapsedTime << " ms" << std::endl;  
    // Elapsed time: 0.0248 ms
    // ##########################################
  
    // 清理  
    cudaEventDestroy(start);  
    cudaEventDestroy(stop);  
    cudaFree(d_a);  
    free(h_a);  
  
    return 0;  
}