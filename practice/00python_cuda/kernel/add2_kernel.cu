__global__ void add2_kernel(float* c,
    const float* a,
    const float* b,
    int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
        i < n; i += gridDim.x * blockDim.x) {
            c[i] = a[i] + b[i];
        }
}

// 函数实现
// add2_kernel是kernel函数，运行在GPU端的。
// 而launch_add2是CPU端的执行函数，调用kernel。注意它是异步的，调用完之后控制权立刻返回给CPU，所以之后计算时间的时候要格外小心，很容易只统计到调用的时间。
void launch_add2(float* c,
const float* a,
const float* b,
int n) {
dim3 grid((n + 1023) / 1024);
dim3 block(1024);
add2_kernel<<<grid, block>>>(c, a, b, n); // 调用函数add2_kernel
}