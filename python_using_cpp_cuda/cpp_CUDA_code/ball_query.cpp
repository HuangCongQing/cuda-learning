#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "ball_query_gpu.h"

extern THCState *state;

#define CHECK_CUDA(x) do { \
	  if (!x.type().is_cuda()) { \
		      fprintf(stderr, "%s must be CUDA tensor at %s:%d\n", #x, __FILE__, __LINE__); \
		      exit(-1); \
		    } \
} while (0)
#define CHECK_CONTIGUOUS(x) do { \
	  if (!x.is_contiguous()) { \
		      fprintf(stderr, "%s must be contiguous tensor at %s:%d\n", #x, __FILE__, __LINE__); \
		      exit(-1); \
		    } \
} while (0)
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

int ball_query_wrapper_cpp(int b, int n, int m, float radius, int nsample, 
    at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor idx_tensor) {
    // b：batch，n为输入点集P的个数，m是输出中心点的个数，radius是每个球的半径，nsample是每个球内最多允许找的点数，new_xyz_tensor[b,m,3]代表中心点坐标，xyz_tensor[b,n,3]代表原始点集合P的坐标，
    // idx_tensor[b,m,nsample]为输出的m个中心点聚合的nsample个点的idx

    // 检查输入是否为contiguous的torch.cuda变量
    CHECK_INPUT(new_xyz_tensor);
    CHECK_INPUT(xyz_tensor);

    // 建立指针
    const float *new_xyz = new_xyz_tensor.data<float>();
    const float *xyz = xyz_tensor.data<float>();
    int *idx = idx_tensor.data<int>();

    // 放入到CUDA中进行具体的算法实现
    ball_query_kernel_launcher_cuda(b, n, m, radius, nsample, new_xyz, xyz, idx);
    return 1;
}
