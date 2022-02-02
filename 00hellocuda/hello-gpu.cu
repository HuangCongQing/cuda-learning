/*
 * @Description: 
 * @Author: HCQ
 * @Company(School): UCAS
 * @Email: 1756260160@qq.com
 * @Date: 2022-02-02 20:13:18
 * @LastEditTime: 2022-02-02 20:39:35
 * @FilePath: /cuda-learning/00hellocuda/hello-gpu.cu
 */
#include <stdio.h>

void cpu(){
    printf("hello cpu\n");
}
// global
__global__ void gpu(){
    printf("hello gpu\n");
}


int main(){
    cpu();
    gpu<<<1,1>>>();  // gpu配置
    cudaDeviceSynchronize();
}