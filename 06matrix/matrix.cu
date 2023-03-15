#include <stdio.h>

#define N 64 // 定义

// Kernal函数
__global__ void gpu(int *a, int *b, int *c_gpu){
    int r = blockDim.x * blockIdx.x + threadIdx.x;
    int c = blockDim.y * blockIdx.y + threadIdx.y;

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
    // 1 初始化
    for(int r=0;r<N;r++){
        for(int c=0;c<N;c++){
            a[r * N + c]  = r;
            b[r * N + c]  = c;
            c_cpu[r * N + c]  = 0;
            c_gpu[r * N + c]  = 0;
        }
    }
    // 2 定义threads
    dim3 threads(16, 16, 1); // 1维写成3维
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y, 1 );
    // 调用
    gpu<<<blocks, threads>>>(a, b, c_gpu);
    cudaDeviceSynchronize();

    cpu(a, b, c_cpu);
    check(a, b, c_cpu, c_gpu) ? printf("ok") : printf("error");

    // end: 释放内存
    cudaFree(a); // 提前写，防止忘记释放内存
    cudaFree(b); 
    cudaFree(c_cpu); 
    cudaFree(c_gpu); 
    return 0;
}