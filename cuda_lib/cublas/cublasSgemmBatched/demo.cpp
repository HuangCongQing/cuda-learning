/*
 * @Description: https://www.yuque.com/huangzhongqing/hpc/fgcw3kuaa8qfgvwi#QgRXi
 * @Author: HCQ
 * @Company(School): UCAS
 * @Email: 1756260160@qq.com
 * @Date: 2024-07-03 18:22:44
 * @LastEditTime: 2024-07-04 09:50:35
 * @FilePath: /cuda-learning/cuda_lib/cublas/cublasSgemmBatched/demo.cpp
 */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <stdio.h>

using namespace std;


__global__ void show(float* ptr, int size)
{
        for(int i =0; i<size; i++)
        printf("%f\n", ptr[i]);
}

int main()
{

        float* a = new float[16];
        for(int i=0; i<16; i++) a[i] = 1.0;

        float* b = new float[32];
        for(int i=0; i<32; i++) b[i] = i+1;

        float* c = new float[16];
        for(int i=0; i<16; i++) c[i] = 3.0;

        float* d_a, *d_b, *d_c;
        size_t size = sizeof(float) * 16;
        cudaMalloc(&d_a, size);
        cudaMalloc(&d_b, size*2);
        cudaMalloc(&d_c, size);

        cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, size*2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice);


        cublasHandle_t handle;
        cublasStatus_t ret;
        ret = cublasCreate(&handle);
        float *a_array[8], *b_array[8];
        float *c_array[8];
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 4; ++j) {
                a_array[i*4+j] = d_a + i * 8 + j * 2;
                b_array[i*4+j] = d_b + i * 2 * 8  + j * 2;
                c_array[i*4+j] = d_c + i * 8 + j * 2;
            }
        }
        const float **d_Marray, **d_Narray;
        float **d_Parray;
        cudaMalloc((void**)&d_Marray, 8*sizeof(float *));
        cudaMalloc((void**)&d_Narray, 16*sizeof(float *));
        cudaMalloc((void**)&d_Parray, 8*sizeof(float *));
        cudaMemcpy(d_Marray, a_array, 8*sizeof(float *), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Narray, b_array, 16*sizeof(float *), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Parray, c_array, 8*sizeof(float *), cudaMemcpyHostToDevice);


        const float alpha  =  1.0f;
        const float beta  =  0.0f;
        int m = 2;
        int n = 1;
        int k = 2;
        int lda = 8;
        int ldb = 8;
        int ldc = 8;
        int batch = 8;

        // 函数==========================================================
       ret = cublasSgemmBatched(handle,
                           CUBLAS_OP_N,
                           CUBLAS_OP_N,
                           m,n,k,
                           &alpha,
                           d_Narray,  ldb,
                           d_Marray,  lda,
                           &beta,
                           d_Parray,  ldc,
                           batch);
        cublasDestroy(handle);
        if (ret == CUBLAS_STATUS_SUCCESS)
        {
        printf("sgemm success  %d, line(%d)\n", ret, __LINE__);
        }

        // show<<<1, 1>>>(c_array[0], 16);
        cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
        for(int i=0; i<16; i++) cout<<c[i]<<" "<<endl;


        return 0;
}

