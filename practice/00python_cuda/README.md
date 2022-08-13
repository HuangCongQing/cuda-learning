## PyTorch是如何调用自定义的CUDA算子

参考：https://github.com/godweiyang/NN-CUDA-Example



## Compare kernel running time
```
python3 pytorch/time.py --compiler jit
python3 pytorch/time.py --compiler setup
python3 pytorch/time.py --compiler cmake
```

## Train model
```
python3 pytorch/train.py --compiler jit
python3 pytorch/train.py --compiler setup
python3 pytorch/train.py --compiler cmake
```