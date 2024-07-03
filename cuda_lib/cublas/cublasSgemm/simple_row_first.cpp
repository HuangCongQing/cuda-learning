/*
 * @Description: // CUBLAS_OP_T （转置 行优先） https://www.yuque.com/huangzhongqing/hpc/deffu5up15g9mqum#P6Itc
 * @Author: HCQ
 * @Company(School): UCAS
 * @Email: 1756260160@qq.com
 * @Date: 2024-07-03 17:16:14
 * @LastEditTime: 2024-07-03 17:23:30
 * @FilePath: /cuda-learning/cuda_lib/cublas/cublasSgemm/simple_row_first.cpp
 */
#include <assert.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
 
 
using namespace std;
 
 
//cuBLAS代码
int main()
{
const float alpha = 1.0f;
const float beta  = 0.0f;
int m = 2, n = 4, k = 3;
 
 
float A[6] = {1,2,3,4,5,6};
float B[12] = {1,2,3,4,5,6,7,8,9,10,11,12};
float *C;
float* d_A,*d_B, *d_C;
C = (float*)malloc(sizeof(float)*8);
cudaMalloc((void**)&d_A, sizeof(float)*6);
cudaMalloc((void**)&d_B, sizeof(float)*12);
cudaMalloc((void**)&d_C, sizeof(float)*8);
 
 
cudaMemcpy(d_A, A, 6*sizeof(float),  cudaMemcpyHostToDevice);
cudaMemcpy(d_B, B, 12*sizeof(float), cudaMemcpyHostToDevice);
 
cublasHandle_t handle;
cublasCreate(&handle);
// CUBLAS_OP_T （转置 行优先）
cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &alpha, d_A, 3, d_B, 4, &beta, d_C, 2);
cublasDestroy(handle);
 
cudaMemcpy(C, d_C, 8*sizeof(float), cudaMemcpyDeviceToHost);
 
for(int i=0; i<8; i++)
{
   cout<<C[i]<<endl;
}
/*  
Output:
38
83
44
98
50
113
56
128
*/
 
}