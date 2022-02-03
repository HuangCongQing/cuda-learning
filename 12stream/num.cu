#include <stdio.h>

#define N 10

__global__ void gpu(int num){
    printf("%d \n", num);
}
int main(){
    for(int i=0;i<N;i++){
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        gpu<<<1,1, 0, stream>>>(i);
        cudaStreamDestroy(stream);
    }
    cudaDeviceSynchronize(); // must
}