
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#include <cmath>
#include <iostream>

__global__ void softmax(float* input, float* output, int size)

{
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  if (index < size) {
    // 解出input中的最大值max_val
    float max_val = input[index];
    for (int i = 0; i < size; i++) {
      max_val = fmax(max_val, input[i]);
    }
    // 使用max_val对input中的元素进行归一化（防止计算指数以及指数和时数据溢出），再计算每一个元素的指数值；
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
      sum += exp(input[i] - max_val);
    }
    // 最后计算每一个元素对应指数的指数占比（概率）；
    output[index] = exp(input[index] - max_val) / sum;

    // only max============
    // float max_val = 0.0f;
    // int max_idx = -1;
    // float sum = 0.0f;
    // for (int i = 0; i < size; i++) {
    //     const float logits_exp = exp(input[i])
    //     sum +=logits_exp;
    //     if(logits_exp > max_val){
    //         max_val = logits_exp;
    //         max_idx = i;
    //     }
    // }
    // float max_prob = max_val;
    // int label = max_idx;
  }
}

// Host code to invoke the kernel
void softmax_wrapper(float* h_input, float* h_output, int size) {
  float *d_input, *d_output;
  cudaMalloc((void**)&d_input, size * sizeof(float));
  cudaMalloc((void**)&d_output, size * sizeof(float));
  cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;
  softmax<<<gridSize, blockSize>>>(d_input, d_output, size);
  cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_input);
  cudaFree(d_output);
}
