#include <stdio.h>

#define N 10

__global__ void gpu(int num){
    printf("%d \n", num);
}
int main(){
    for(int i=0;i<N;i++){
        cudaStream_t stream;
        cudaStreamCreate(&stream); //创建流
        gpu<<<1,1, 0, stream>>>(i); // 作为参数传进核函数
        cudaStreamDestroy(stream); // 销毁流
    }
    cudaDeviceSynchronize(); // must
}