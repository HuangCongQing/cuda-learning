'''
Description: 矩阵转置numba版本
    a = cuda.to_device(a) # cudaMemcpy HTOD
    cuda.synchronize()
    a.copy_to_host() # DTOH
    a_out = cuda.device_array_like(a_in)
    cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    tile = cuda.shared.array((32, 32), numba.types.int32)
    cuda.syncthreads()
    func[grid_size, block_size](d_a, d_b, d_c)

Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2022-02-04 23:11:09
LastEditTime: 2022-02-04 23:13:12
FilePath: /cuda-learning/16numba/transpose.py
'''
""" 
from numba import cuda

@cuda.jit
def func(x, y):
    pass
"""

from numba import cuda
import numba
import numpy as np

MX = 20480
MY = 20480
TILE_DIM = 32
BLOCK_SIZE = 8

@cuda.jit
def transpose(odata, idata):
     tile = cuda.shared.array((TILE_DIM, TILE_DIM), numba.types.float32)
     x = cuda.blockIdx.x  * TILE_DIM + cuda.threadIdx.x
     y = cuda.blockIdx.y  * TILE_DIM + cuda.threadIdx.y
     w = cuda.gridDim.x * TILE_DIM

     if x >= MX or y >=MY: return

     for i in range(0, TILE_DIM, BLOCK_SIZE):
         tile[cuda.threadIdx.y + i, cuda.threadIdx.x] = idata[y + i, x]

     cuda.synchronize()
    # 转置
     x = cuda.blockIdx.y  * TILE_DIM + cuda.threadIdx.y
     y = cuda.blockIdx.x  * TILE_DIM + cuda.threadIdx.x
     for i in range(0, TILE_DIM, BLOCK_SIZE):
        #  tile[cuda.threadIdx.y + i, cuda.threadIdx.x] = idata[y + i, x]
        odata[y + i, x0] = tile[cuda.threadIdx.x, cuda.threadIdx.y, y+i]

threads = (TILE_DIM, BLOCK_SIZE)
blocks = (MX + TILE_DIM + 1) // TILE_DIM,   (MY + TILE_DIM + 1) // TILE_DIM
a_in = cuda.to_device(np.arange(MX*MY, dtype=np.float32).reshape((MX, MY)))
a_out = cuda.device_array_like(a_in)
%timeit transpose[blocks, threads](a_in, a_out); cuda.synchronize()