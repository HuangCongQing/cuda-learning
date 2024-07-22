// Doc: https://www.yuque.com/huangzhongqing/cuda/mdsl5etmw4sxl8bo
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <iostream> // std::cout

#define N 32 * 1024 * 1024  // 4M
#define BLOCK_SIZE 256  // BLOCK_SIZE = 256， 也就是一个 Block 有 8 个 warp

//  g_idata 表示的是输入数据的指针，而 g_odata 则表示输出数据的指针。
// 然后首先把 global memory 数据 load 到 shared memory 中，接着在 shared memory 中对数据进行 Reduce Sum 操作，
// 最后将 Reduce Sum 的结果写会 global memory 中
__global__ void reduce_v0(float *g_idata, float *g_odata) {
  __shared__ float sdata[BLOCK_SIZE]; // 共享内存

  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = g_idata[i]; //赋值
  __syncthreads();

  // do reduction in shared mem（求和）
  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int main() {
  // 1 申请内存（host device）
  float *input_host = (float *)malloc(N * sizeof(float));
  float *input_device;
  cudaMalloc((void **)&input_device, N * sizeof(float));
  
  float *output_host = (float *)malloc((N / BLOCK_SIZE) * sizeof(float));
  float *output_device;
  cudaMalloc((void **)&output_device, (N / BLOCK_SIZE) * sizeof(float));
  // 赋值
  for (int i = 0; i < N; i++) {
    input_host[i] = 2.0;
  }
  // 2 将host数据拷贝到device
  cudaMemcpy(input_device, input_host, N * sizeof(float),
             cudaMemcpyHostToDevice);

  // 3 执行GPU函数
  dim3 grid(N / BLOCK_SIZE, 1);
  dim3 block(BLOCK_SIZE, 1);
  reduce_v0<<<grid, block>>>(input_device, output_device);
  // 4 device数据拷贝到host
  int32_t block_num = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  cudaMemcpy(output_device, output_host, block_num * sizeof(float),
             cudaMemcpyDeviceToHost);
  // 输出
  std::cout<<"输出："<<output_host[0]<<std::endl;


  return 0;
}
