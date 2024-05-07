/*
 * @Description: https://blog.csdn.net/u011197534/article/details/78378536
 * @Author: HCQ
 * @Company(School): UCAS
 * @Email: 1756260160@qq.com
 * @Date: 2024-05-07 19:45:29
 * @LastEditTime: 2024-05-07 19:46:17
 * @FilePath: /cuda-learning/cuda_lib/cublas/cublasSgemm/row_first.cpp
 * 编译运行：nvcc row_first.cpp -o row_first -L/usr/local/cuda/lib64 -lcudart -lcuda -lcublas 
 */
// CUDA runtime 库 + CUBLAS 库
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <iostream>
#include <stdlib.h>

using namespace std;

// 定义测试矩阵的维度
int const A_ROW = 5;
int const A_COL = 6;
int const B_ROW = 6;
int const B_COL = 7;

int main()
{
  // 定义状态变量
  cublasStatus_t status;
  float *h_A,*h_B,*h_C;   //存储于内存中的矩阵
  h_A = (float*)malloc(sizeof(float)*A_ROW*A_COL);  //在内存中开辟空间
  h_B = (float*)malloc(sizeof(float)*B_ROW*B_COL);
  h_C = (float*)malloc(sizeof(float)*A_ROW*B_COL);

  // 为待运算矩阵的元素赋予 0-10 范围内的随机数
  for (int i=0; i<A_ROW*A_COL; i++) {
    h_A[i] = (float)(rand()%10+1);
  }
  for(int i=0;i<B_ROW*B_COL; i++) {
    h_B[i] = (float)(rand()%10+1);
  }
  // 打印待测试的矩阵
  cout << "矩阵 A :" << endl;
  for (int i=0; i<A_ROW*A_COL; i++){
    cout << h_A[i] << " ";
    if ((i+1)%A_COL == 0) cout << endl;
  }
  cout << endl;
  cout << "矩阵 B :" << endl;
  for (int i=0; i<B_ROW*B_COL; i++){
    cout << h_B[i] << " ";
    if ((i+1)%B_COL == 0) cout << endl;
  }
  cout << endl;

  float *d_A,*d_B,*d_C;    //存储于显存中的矩阵
  cudaMalloc((void**)&d_A,sizeof(float)*A_ROW*A_COL); //在显存中开辟空间
  cudaMalloc((void**)&d_B,sizeof(float)*B_ROW*B_COL);
  cudaMalloc((void**)&d_C,sizeof(float)*A_ROW*B_COL);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cudaMemcpy(d_A,h_A,sizeof(float)*A_ROW*A_COL,cudaMemcpyHostToDevice); //数据从内存拷贝到显存
  cudaMemcpy(d_B,h_B,sizeof(float)*B_ROW*B_COL,cudaMemcpyHostToDevice);

  float a = 1, b = 0;
  cublasSgemm(
          handle,
          CUBLAS_OP_T,   //矩阵A的属性参数，转置，按行优先
          CUBLAS_OP_T,   //矩阵B的属性参数，转置，按行优先
          A_ROW,          //矩阵A、C的行数
          B_COL,          //矩阵B、C的列数
          A_COL,          //A的列数，B的行数，此处也可为B_ROW,一样的
          &a,             //alpha的值
          d_A,            //左矩阵，为A
          A_COL,          //A的leading dimension，此时选择转置，按行优先，则leading dimension为A的列数
          d_B,            //右矩阵，为B
          B_COL,          //B的leading dimension，此时选择转置，按行优先，则leading dimension为B的列数
          &b,             //beta的值
          d_C,            //结果矩阵C
          A_ROW           //C的leading dimension，C矩阵一定按列优先，则leading dimension为C的行数
  );
  //此时得到的结果便是C=AB,但由于C是按列优先，故此时得到的C应该是正确结果的转置
  std::cout << "计算结果的转置 ( (A*B)的转置 )：" << std::endl;


  cudaMemcpy(h_C,d_C,sizeof(float)*A_ROW*B_COL,cudaMemcpyDeviceToHost);
  for(int i=0;i<A_ROW*B_COL;++i) {
    std::cout<<h_C[i]<<" ";
    if((i+1)%B_COL==0) std::cout<<std::endl;
  }
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(h_C);
  return 0;
}

/*  
矩阵 A :
4 7 8 6 4 6 
7 3 10 2 3 8 
1 10 4 7 1 7 
3 7 2 9 8 10 
3 1 3 4 8 6 

矩阵 B :
10 3 3 9 10 8 4 
7 2 3 10 4 2 10 
5 8 9 5 6 1 4 
7 2 1 7 4 3 1 
7 2 6 6 5 8 7 
6 7 10 4 8 5 6 

计算结果的转置 ( (A*B)的转置 )：
235 224 198 268 172 152 173 
120 143 101 195 220 152 205 
151 236 207 212 258 152 208 
229 163 226 156 134 142 96 
181 135 188 169 176 215 130 
*/