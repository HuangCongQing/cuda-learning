/*
 * @Description: 
 * @Author: HCQ
 * @Company(School): UCAS
 * @Email: 1756260160@qq.com
 * @Date: 2022-02-02 20:13:18
 * @LastEditTime: 2022-02-02 20:54:06
 * @FilePath: /cuda-learning/00hellocuda/hello-gpu.cu
 */
#include <stdio.h>

void cpu(){
    printf("hello cpu\n");
}
// global 将在gpu上运行并可全局调用
__global__ void gpu(){
    printf("hello gpu\n");
}


int main(){
    cpu();
    gpu<<<1,1>>>();  // gpu配置 《线程 线程块个数》
    cudaDeviceSynchronize();
}