
// 分配空间函数不能放在核函数里面,放在外面. 
// cudaMallocManaged(&a, size); // 分配内存，既可以被cpu使用也可以被gpu使用
// cudaFree(a);  // 释放内存
// https://www.yuque.com/huangzhongqing/cuda/nuqxif

#include <stdio.h>
#include <stdlib.h> // cpu的malloc函数

void cpu(int *a , int N){
    for(int i=0;i<N;i++){
        a[i] = i;
    }
    printf("hello cpu\n");
}


// 函数定义:定义一个数组,把数组内的值放大两倍.<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
// global 将在gpu上运行并可全局调用
__global__ void gpu(int *a, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x; // 类似for循环了!
    if(i<N){ // 数组大小是N，需要判断下
        a[i]*=2; // 放大2倍
    }
}

// 验证
bool check(int *a, int N){
    for(int i = 0;i<N;i++){
        if(a[i] != 2*i) return false;
    }
    return true;
}


int main(){
    const int N = 2 << 5; //二进制左移运算符。2的5次方 多少个数
    size_t size = N*sizeof(int); //内存 nBytes = N * sizeof(float);
    int *a; //取指针的地址&a
    cudaMallocManaged(&a, size); //  给a分配的内存.  cudaMallocManaged既可以被cpu使用也可以被gpu使用
    // cpu
    cpu(a, N);
    // gpu
    size_t threads = 256; // 定义线程数量
    size_t blocks = (N + threads  -1)/threads; //blocks 希望每一个数都拥有一个线程  | 算法竞赛向上取整,等同于ceil函数
    gpu<<<blocks, threads>>>(a, N); // 
    cudaDeviceSynchronize(); // 同步一下,不然报错!！！！！！！！！！！！！！！！！

    check(a, N)?printf("Ok") : printf("Sorry");
    cudaFree(a);
}