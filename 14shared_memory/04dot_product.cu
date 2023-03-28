// 点乘（Dot Product） with 共享内存  https://www.yuque.com/huangzhongqing/cuda/gln2mn#dkDJQ
#include <stdio.h>

// 核函数
__global__ void dot_plus_1D(float* a, float* b, float* c, float length){
    // 申请一个shared memory，用于缓存中间结果。
    // 需要注意的是，在申请的时候不能使用动态申请，大小需要时已知常量。 256是block内thread的大小，保证了每个线程的临时变量都能顺利存储。
    __shared__ float cache[256]; // 

    int x = threadIdx.x + blockIdx.x * blockDim.x;
//    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int curr_index = threadIdx.x;
    float temp = 0;

    if (x < length){
        temp += a[x] * b[x];

    }
    cache[curr_index] = temp;

    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0){
        if (curr_index < i){
            cache[curr_index] += cache[curr_index + i];
        }
        __syncthreads(); 	// 归约的每一步，也需要等前面一步全部执行完成
        i /= 2;
    }

    // 最后在0号线程上，把当前block的结果返回。
    if (curr_index == 0){
        c[blockIdx.x] = cache[0];
    }
}


void CUDA_Base::matrix_dot_plus(float *a, float *b, int length) {
    cudaSetDeviceFlags(cudaDeviceMapHost);

    float *dev_src1, *dev_src2, *dev_dst, *dst, *src1, *src2;
    int byte_size = length * sizeof(float);

    cudaHostAlloc((void**)&src1, byte_size, cudaHostAllocWriteCombined | cudaHostAllocMapped);
    cudaHostAlloc((void**)&src2, byte_size, cudaHostAllocWriteCombined | cudaHostAllocMapped);
    memcpy((void*)src1, (void*)a, byte_size);
    memcpy((void*)src2, (void*)b, byte_size);
    std::cout << src1[2] <<std::endl;
    cudaHostGetDevicePointer(&dev_src1, src1, 0);
    cudaHostGetDevicePointer(&dev_src2, src2, 0);


    // 256
    dim3 block_size(256);
    dim3 grid_size((length / block_size.x) + 1);
    cudaHostAlloc((void**)&dst, grid_size.x * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped);
    cudaHostGetDevicePointer(&dev_dst, dst, 0);

    // dot_plus_1D核函数<<<<<<<<<<<<<<<<<<<<<<<
    dot_plus_1D<<<grid_size, block_size>>>(dev_src1, dev_src2, dev_dst, length);

//    cudaThreadSynchronize();
    cudaDeviceSynchronize();

    float num = 0;
    for (int i = 0; i < grid_size.x; ++i) {
        num += dst[i];
    }

    std::cout << num << std::endl;

    cudaFreeHost(src1);
    cudaFreeHost(src2);
    cudaFreeHost(dst);
}
