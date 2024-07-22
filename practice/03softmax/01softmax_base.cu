
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#include <cmath>
#include <iostream>

template <typename Type>
__global__ void PrintGpuDataKernel(const Type* data, int start, int end) {
  for (int i = start; i < end; ++i) {
    if (std::is_integral<Type>::value) { // 是否是整数类型
      printf("| %.1f ", static_cast<float>(data[i]));
    } else {
      printf("| %.7f ", static_cast<float>(data[i]));
    }
  }
  printf("\n");
}

// PrintGpuDataKernel<Type><<<1, 1>>>(data, start, end);

template <typename Type>
void PrintGpu(const Type* data, int start, int end) {
  PrintGpuDataKernel<Type><<<1, 1>>>(data, start, end);
}

__global__ void softmax(float* input, float* output, int size)

{
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  // 每12个
  const float* source_ptr = input + index * 12;

  if (index < size) {
    // 解出input中的最大值max_val
    float max_val = source_ptr[index];
    for (int i = 0; i < size; i++) {
      max_val = fmax(max_val, source_ptr[i]);
    }
    // 使用max_val对source_ptr中的元素进行归一化（防止计算指数以及指数和时数据溢出），再计算每一个元素的指数值；
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
      sum += exp(source_ptr[i] - max_val);
    }
    // 最后计算每一个元素对应指数的指数占比（概率）；
    output[index] = exp(source_ptr[index] - max_val) / sum;

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
  // 查看输出结果
  PrintGpu<float>(d_output, 0, 12);
  cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_input);
  cudaFree(d_output);
}

int main() {
  int size = 12;

  // 1 申请host内存
  float *h_input, *h_output;
  h_input = (float*)malloc(size * sizeof(float));
  h_output = (float*)malloc(size * sizeof(float));
  // 初始化
  for (int i = 0; i < size; i++) {
    h_input[i] = 1.0;
  }
  // 2 申请GPU内存
  float *d_input, *d_output;
  cudaMalloc((void**)&d_input, size*sizeof(float));
  cudaMalloc((void**)&d_output, size*sizeof(float));

  // 3 拷贝 & GPU
  softmax_wrapper(h_input, h_output, size);

}