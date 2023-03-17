#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ball_query_gpu.h"
#include "cuda_utils.h"


// CUDA使用__global__来定义kernel
__global__ void ball_query_kernel_cuda(int b, int n, int m, float radius, int nsample, 
    const float *__restrict__ new_xyz, const float *__restrict__ xyz, int *__restrict__ idx) {
    // threadIdx是一个三维的向量，可以用.x .y .z分别调用其三个维度。此处我们只初始化了第一个维度为THREADS_PER_BLOCK
    // blockIdx也是三维向量。我们初始化用的DIVUP(m, THREADS_PER_BLOCK), b分别对应blockIdx.x和blockIdx.y
    // blockDim代表block的长度
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= m) return;

    // 针对指针数据，利用+的操作来确定数组首地址，相当于取new_xyz[bi,ni]
    new_xyz += bs_idx * m * 3 + pt_idx * 3;
    xyz += bs_idx * n * 3;
    idx += bs_idx * m * nsample + pt_idx * nsample;

    float radius2 = radius * radius;
    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];

    int cnt = 0;
    for (int k = 0; k < n; ++k) {
        float x = xyz[k * 3 + 0];
        float y = xyz[k * 3 + 1];
        float z = xyz[k * 3 + 2];
        // 算法很简单，循环一遍所有数据算距离
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
        if (d2 < radius2){
            if (cnt == 0){
                for (int l = 0; l < nsample; ++l) {
                    idx[l] = k;
                }
            }
            idx[cnt] = k;
            ++cnt;
            if (cnt >= nsample) break;
        }
    }
}


void ball_query_kernel_launcher_cuda(int b, int n, int m, float radius, int nsample, \
    const float *new_xyz, const float *xyz, int *idx) {

    // cudaError_t变量用来记录CUDA的err信息，在最后需要check
    cudaError_t err;
    // divup定义在cuda_utils.h,DIVUP(m, t)相当于把m个点平均划分给t个block中的线程，每个block可以处理THREADS_PER_BLOCK个线程。
    // THREADS_PER_BLOCK=256，假设我有m=1024个点，那就是我需要4个block，一共256*4个线程去处理这1024个点。
    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    // 可函数需要用<<<blocks, threads>>> 去指定调用的块数和线程数，总共调用的线程数为blocks*threads
    ball_query_kernel_cuda<<<blocks, threads>>>(b, n, m, radius, nsample, new_xyz, xyz, idx);

    // 如果cuda操作错误，则打印错误信息
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
