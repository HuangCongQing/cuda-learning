/* 
基于metirx修改 /home/hcq/github/cuda-learning/05error/error.cu
cudaMallocP: gpu
cudaMallocHost:cpu

● 无论是从主机到设备还是从设备到主机，cudaMemcpy命令均可拷贝（而非传输）内存。
  ○ cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);
  ○ cudaMemcpy(host_a, device_a, size, cudaMemcpyDeviceToHost);
个人笔记: https://www.yuque.com/huangzhongqing/hpc/nuqxif
*/
#include <stdio.h>
#include <assert.h>

#define N 64 // 定义

// Kernal函数
__global__ void gpu(int *a, int *b, int *c_gpu){
    int r = blockDim.x * blockIdx.x + threadIdx.x;
    int c = blockDim.y * blockIdx.y + threadIdx.y;

    if(r < N && c < N){
        c_gpu[r * N + c] = a[r * N + c] + b[r * N +c];
    }
    // printf("hello gpu\n");
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
bool check(int *c_cpu, int * c_gpu){
    for(int r = 0;r < N; r++){
        for(int c = 0;c < N; c++){
            if(c_cpu[r * N +c] != c_gpu[r * N + c]){
                return false;
            }
        }
    }
    return true;
}
// 处理错误的宏定义
inline cudaError_t checkCuda(cudaError_t result){
    if(result !=cudaSuccess){
        fprintf(stderr, "CUDA runtime error: %s\n", cudaGetErrorString(result));
        assert(result==cudaSuccess);
    }
    return result;
}

int main(){
    int *a_cpu, *b_cpu, *a_gpu, *b_gpu, *c_cpu, *c_gpu, *c_gpu_cpu; // // gpu->>cpu===============================================
    size_t size = N * N * sizeof(int);

    cudaMallocHost(&a_cpu, size);
    cudaMallocHost(&b_cpu, size);
    cudaMallocHost(&c_cpu, size);
    cudaMallocHost(&c_gpu_cpu, size); // // gpu->>cpu===============================================
    cudaMalloc(&a_gpu, size); // 此函数 不能在cpu上访问gpu显存
    cudaMalloc(&b_gpu, size); // 此函数 不能在cpu上访问gpu显存
    cudaMalloc(&c_gpu, size); // 此函数 不能在cpu上访问gpu显存
    // 1 初始化
    for(int r=0;r<N;r++){
        for(int c=0;c<N;c++){
            a_cpu[r * N + c]  = r;
            b_cpu[r * N + c]  = c;
            // c_gpu[r * N + c]  = 0; //尝试访问会报错，不能在cpu上访问gpu显存
            c_gpu_cpu[r * N + c]  = 0; // // gpu->>cpu===============================================
            c_cpu[r * N + c]  = 0;
        }
    }
    cpu(a_cpu, b_cpu, c_cpu); // 现在cpu上执行再在gpu上执行

    // 2 定义threads
    dim3 threads(16, 16, 1); // 1维写成3维
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y, 1 );
    // 调用gpu之前cpu转gpu
    cudaMemcpy(a_gpu, a_cpu, size, cudaMemcpyHostToDevice); // cpu->>gpu===============================================
     cudaMemcpy(b_gpu, b_cpu, size, cudaMemcpyHostToDevice); // cpu->>gpu===============================================
    gpu<<<blocks, threads>>>(a_gpu, b_gpu, c_gpu);
    cudaDeviceSynchronize();
    // 拷贝 / gpu->>cpu
    cudaMemcpy(c_gpu_cpu, c_gpu, size, cudaMemcpyDeviceToHost); // gpu->>cpu===============================================
    check(c_cpu, c_gpu_cpu) ? printf("ok") : printf("error");

    // end: 释放内存
    cudaFreeHost(a_cpu); // 提前写，防止忘记释放内存
    cudaFreeHost(b_cpu); 
    cudaFreeHost(c_cpu); 
    cudaFree(c_gpu);  //gpu
    return 0;
}