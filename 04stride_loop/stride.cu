// 跨步循环(数据量比block*threads 要多) https://www.yuque.com/huangzhongqing/hpc/xtl960#SkGBb
#include <stdio.h>
#include <stdlib.h> // cpu的malloc函数

void cpu(int *a , int N){
    for(int i=0;i<N;i++){
        a[i] = i;
    }
    printf("hello cpu\n");
}

// global 将在gpu上运行并可全局调用
__global__ void gpu(int *a, int N){
    int threadi = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x; // (block数量*每个block的线程数)=所有线程数===============================
    for (int i = threadi;i<N;i+=stride){
        a[i]*=2; // 放大2倍
    }

    printf("hello gpu\n");
}

// 验证
bool check(int *a, int N){
    for(int i = 0;i<N;i++){
        if(a[i] != 2*i) return false;
    }
    return true;
}


int main(){
    const int N = 2 << 5; //二进制左移运算符。
    size_t size = N*sizeof(int);
    int *a; //取指针的地址&a
    cudaMallocManaged(&a, size); // 既可以被cpu使用也可以被gpu使用
    cpu(a, N);
    
    // gpu
    size_t threads = 256;
    size_t blocks = (N + threads  -1)/threads; // 算法竞赛向上取整  ceil也可
    gpu<<<blocks, threads>>>(a, N); // 每一个数都拥有一个线程
    cudaDeviceSynchronize();

    check(a, N)?printf("Ok") : printf("Sorry, error");
    cudaFree(a);
}