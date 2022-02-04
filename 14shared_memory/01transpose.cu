// 矩阵转置
#include <stdio.h>

#define TILE_DIM 32
#define BLOCK_SIZE 8
#define MX 2048
#define MY 2048 

__global__ void transpose(float* odata, float* idata){
    int x = blockIdx.x * TILE_DIM + threadIdx.x; // 一个block里面分TILE_DIM*TILE_DIM，每个TILE_DIM厘米有thread线程数量
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    int w = gridDim.x * TILE_DIM;
    if(x >= MX || y >=MY) return;
    for(int i = 0; i<TILE_DIM; i += BLOCK_SIZE){ // 一个block是一个矩阵中的一个数
        odata[x * w + y + i] = idata[(y + i) * w + x];
    }
}


// 验证
bool check(float *c_cpu, float * c_gpu){
    for(int r = 0;r < MX; r++){
        for(int c = 0;c < MY; c++){
            if(c_cpu[r * MX +c] != c_gpu[r * MY + c]){
                return false;
            }
        }
    }
    return true;
}

int main(){
    size_t size = MX * MY * sizeof(float);
    float *h_idata, *h_odata, *d_idata, *d_odata, *res; // host device
    cudaMallocHost(&h_idata, size);
    cudaMallocHost(&h_odata, size); 
    cudaMallocHost(&res, size); 
    cudaMalloc(&d_idata, size);
    cudaMalloc(&d_odata, size);

    // 2 定义threads
    dim3 threads(TILE_DIM, BLOCK_SIZE, 1); // 1维写成3维
    dim3 blocks((MX + TILE_DIM - 1) / TILE_DIM, (MY + TILE_DIM - 1) / TILE_DIM, 1 );
    // c初始化
    for(int i = 0; i < MX;i++){
        for(int j = 0; j < MY; j++){
            h_idata[i * MY + j] = i * MY + j;
            res[i * MY + j] = j * MY + i;
        }
    }
    
    cudaMemcpy(d_idata, h_idata, size, cudaMemcpyHostToDevice); // cpu->>gpu
    // 多跑几轮
    // for(int i = 0; i < 100; i++)
    transpose<<<blocks, threads>>>(d_odata, d_idata);
    cudaMemcpy(h_odata, d_odata, size, cudaMemcpyDeviceToHost); // gpu->>cpu
    cudaDeviceSynchronize();
    check(res, h_odata) ? printf("ok") : printf("error");

    // end: 释放内存
    cudaFreeHost(h_idata); // 提前写，防止忘记释放内存
    cudaFreeHost(h_odata); 
    cudaFreeHost(res); 
    cudaFree(d_idata);  //gpu
    cudaFree(d_odata);  //gpu


}