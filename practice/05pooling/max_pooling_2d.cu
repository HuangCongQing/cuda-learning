#include <iostream>

__global__ void MaxPool2d(float* bottom_data, const int height, const int width,
                          const int pooled_height, const int out_height,
                          float* top_data) {
  int x = blockIdx.x;
  int y = blockIdx.y;
  int i, j, u, v, index;
  int index2 =
      x * gridDim.y * out_height * out_height + y * out_height * out_height;
  float s;
  for (i = 0; i < out_height; ++i)
    for (j = 0; j < out_height; ++j) {
      index = x * gridDim.y * height * width + y * height * width +
              i * pooled_height * width + j * pooled_height;
      s = -10000.0;
      for (u = 0; u < pooled_height && (u + pooled_height * i) < height; ++u)
        for (v = 0; v < pooled_height && (v + pooled_height * j) < width; ++v)
          if (*(bottom_data + index + u * width + v) > s)
            s = *(bottom_data + index + u * width + v);
      *(top_data + index2) = s;
      ++index2;
    }
}

int main() {
  const int N = 500, M = 100, H = 24, W = 24, D = 2;
  const int PH = H / D + H % D;
  int image_size = N * M * H * W * sizeof(float);
  int out_size = N * M * PH * PH * sizeof(float);
  float mul_by = 0.01;
  float *input, *output, *dev_output, *dev_input;
  input = new float[image_size];
  output = new float[out_size];
  for (int i = 0; i < N * M * H * W; i++) *(input + i) = i * mul_by;

  cudaMalloc((void**)&dev_output, out_size);
  cudaMalloc((void**)&dev_input, image_size);
  cudaMemcpy(dev_input, input, image_size, cudaMemcpyHostToDevice);
  dim3 grid(N, M);
  // 创建CUDA事件
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // 记录开始时间
  cudaEventRecord(start, 0);

  //   kernel函数
  MaxPool2d<<<grid, 1>>>(dev_input, H, W, D, PH, dev_output);
  // 等待核函数完成
  cudaDeviceSynchronize();
  // 记录结束时间
  cudaEventRecord(stop, 0);
  // 等待所有事件都完成，确保可以安全查询时间
  cudaEventSynchronize(stop);
  // 计算并打印时间
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  std::cout << "Elapsed time: " << elapsedTime << " ms" << std::endl;

  cudaMemcpy(output, dev_output, out_size, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 10; i++) std::cout << *(output + i) << std::endl;

  cudaFree(dev_input);
  cudaFree(dev_output);
  delete[] output;
  delete[] input;
}

/*
Cost: 141ms.
0.25
0.27
0.29
0.31
0.33
0.35
0.37
0.39
0.41
0.43
*/