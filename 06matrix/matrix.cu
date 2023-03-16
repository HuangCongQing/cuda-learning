// 实现矩阵加法(两维数据映射到一维坐标上面)
// video: https://www.bilibili.com/video/BV1aq4y1T7Dr/?spm_id_from=333.788&vd_source=617461d43c4542e4c5a3ed54434a0e55
// docs：https://www.yuque.com/huangzhongqing/cuda/em914n

#include <stdio.h> //printf()

#define N 64 // 定义

// Kernal函数
__global__ void gpu(int *a, int *b, int *c_gpu){
    int r = blockDim.x * blockIdx.x + threadIdx.x; // 16
    int c = blockDim.y * blockIdx.y + threadIdx.y; // 16

    if(r < N && c < N){
        c_gpu[r * N + c] = a[r * N + c] + b[r * N +c];
    }
    printf("hello gpu\n");
}

// cpu
void cpu(int *a, int *b, int *c_cpu){
    for(int r = 0;r < N; r++){
        for (int c = 0;c < N; c++){
            c_cpu[r * N +c]  = a[r * N +c] + b[r * N + c];
        }
    }
    printf("hello cpu\n");
}


// 验证
bool check(int *a, int *b, int *c_cpu, int * c_gpu){
    for(int r = 0;r < N; r++){
        for(int c = 0;c < N; c++){
            if(c_cpu[r * N +c] != c_gpu[r * N + c]){
                return false;
            }
        }
    }
    return true;
}

int main(){
    int *a, *b, *c_cpu, *c_gpu;
    size_t size = N * N * sizeof(int);

    cudaMallocManaged(&a, size);  // 统一内存管理cudaMallocManaged既可以被cpu使用也可以被gpu使用
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c_cpu, size);
    cudaMallocManaged(&c_gpu, size);
    // 1 初始化(两维数据映射到一维坐标上面)
    for(int r=0;r<N;r++){
        for(int c=0;c<N;c++){
            a[r * N + c]  = r;
            b[r * N + c]  = c;
            c_cpu[r * N + c]  = 0;
            c_gpu[r * N + c]  = 0;
        }
    }
    // 2 定义threads
    dim3 threads(16, 16, 1); // 三维的（1维表示成3维，方便不同维度去操作 timeline:https://www.bilibili.com/video/BV1aq4y1T7Dr?t=455.9）
    // 跨步操作（64维数据，但是kernel只要16，所有要跨步操作）
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y, 1 );
    // 调用
    gpu<<<blocks, threads>>>(a, b, c_gpu);
    cudaDeviceSynchronize(); // GPU数据同步到cpu上

    cpu(a, b, c_cpu);
    check(a, b, c_cpu, c_gpu) ? printf("ok") : printf("error");

    // end: 释放内存
    cudaFree(a); // 提前写，防止忘记释放内存
    cudaFree(b); 
    cudaFree(c_cpu); 
    cudaFree(c_gpu); 
    return 0;
}