/*
 * @Description: 
 * @Author: HCQ
 * @Company(School): UCAS
 * @Email: 1756260160@qq.com
 * @Date: 2022-02-02 20:13:18
 * @LastEditTime: 2022-02-02 21:18:00
 * @FilePath: /cuda-learning/00hellocuda/hello-gpu.cu
 */
#include <stdio.h>

void cpu(){
    printf("hello cpu\n");
}
// global 将在gpu上运行并可全局调用
__global__ void gpu(){
    // 只希望第一个block的第一个线程去打印
    if(blockIdx.x==0&&threadIdx.x==0){
        printf("hello gpu\n");
    }
}


int main(){
    cpu();
    gpu<<<2,3>>>();  // gpu配置 《block 线程数》
    cudaDeviceSynchronize();
}