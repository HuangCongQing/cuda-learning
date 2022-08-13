import time
import argparse
import numpy as np
import torch

# c = a + b (shape: [n])
n = 1024 * 1024
a = torch.rand(n, device="cuda:0")
b = torch.rand(n, device="cuda:0")
cuda_c = torch.rand(n, device="cuda:0")

ntest = 10

def show_time(func):
    times = list()
    res = None
    # GPU warm up
    for _ in range(10):
        res = func()
    for _ in range(ntest):
        # sync the threads to get accurate cuda running time
        torch.cuda.synchronize(device="cuda:0")
        start_time = time.time()
        func()
        torch.cuda.synchronize(device="cuda:0")
        end_time = time.time()
        times.append((end_time-start_time)*1e6)
    return times, res

# torch_launch_add2是cuda函数名
def run_cuda():
    if args.compiler == 'jit':
        cuda_module.torch_launch_add2(cuda_c, a, b, n) # 封装好的接口来进行调用。
    elif args.compiler == 'setup':
        add2.torch_launch_add2(cuda_c, a, b, n)
    elif args.compiler == 'cmake':
        torch.ops.add2.torch_launch_add2(cuda_c, a, b, n)
    else:
        raise Exception("Type of cuda compiler must be one of jit/setup/cmake.")

    return cuda_c

def run_torch():
    c = a + b
    return c.contiguous()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 三种参数
    parser.add_argument('--compiler', type=str, choices=['jit', 'setup', 'cmake'], default='jit')
    args = parser.parse_args()

    # torch.utils.cpp_extension.load函数就是用来自动编译上面的几个cpp和cu文件的。
    # 主要的就是sources参数，指定了需要编译的文件列表。
    if args.compiler == 'jit':
        from torch.utils.cpp_extension import load
        cuda_module = load(name="add2",
                           extra_include_paths=["include"],
                           sources=["pytorch/add2_ops.cpp", "kernel/add2_kernel.cu"],
                           verbose=True)
    # setup方式
    elif args.compiler == 'setup':
        import add2 # 可以之间import  调用add2.torch_launch_add2(cuda_c, a, b, n)
    # cmake  so文件的方式
    elif args.compiler == 'cmake':
        torch.ops.load_library("build/libadd2.so") # 加载的编译so文件
    else:
        raise Exception("Type of cuda compiler must be one of jit/setup/cmake.")

    print("Running cuda...")
    cuda_time, cuda_res = show_time(run_cuda) # 调用函数
    print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))

    print("Running torch...")
    torch_time, torch_res = show_time(run_torch) # 运行
    print("Torch time:  {:.3f}us".format(np.mean(torch_time)))

    torch.allclose(cuda_res, torch_res)
    print("Kernel test passed.")

    ''' 
python3 pytorch/time.py --compiler setup

    Running cuda...
    Cuda time:  42.892us
    Running torch...
    Torch time:  43.225us
    Kernel test passed.
    
     '''