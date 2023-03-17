
import torch
from torch.autograd import Function

from cpp_CUDA_code import pointnet_cuda as pointnet

if __name__ == '__main__':
    batch_size = 4
    N = 100000
    npoint = 2
    radius = 3
    nsample = 5
    xyz = torch.rand([batch_size, N, 3]) * 100  # 0~100 均匀分布
    new_xyz = torch.rand([batch_size, npoint, 3]) * 100
    idx = torch.cuda.IntTensor(batch_size, npoint, nsample).zero_()
    pointnet.ball_query_wrapper(batch_size, N, npoint, radius, nsample, new_xyz.cuda(), xyz.cuda(), idx)
    print(idx)