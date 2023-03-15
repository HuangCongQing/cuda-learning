// 官方实现：pi的次方再开方(能看到多个核函数重叠)

// https://developer.nvidia.com/blog/gpu-pro-tip-cuda-7-streams-simplify-concurrency/
const int N = 1 << 20;

__global__ void kernel(float *x, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        x[i] = sqrt(pow(3.14159,i)); // pi的次方再开方
    }
}

int main()
{
    const int num_streams = 8;

    cudaStream_t streams[num_streams]; // 自定义stream流的数量
    float *data[num_streams]; // 没有赋值

    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
 
        cudaMalloc(&data[i], N * sizeof(float)); // 分配GPU内存
        
        // launch one worker kernel per stream // 每个流加载一个kernel
        kernel<<<1, 64, 0, streams[i]>>>(data[i], N);

        // launch a dummy kernel on the default stream // 默认流
        kernel<<<1, 1>>>(0, 0);
    }

    cudaDeviceReset();

    return 0;
}