#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "ball_query_gpu.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ball_query_wrapper", &ball_query_wrapper_cpp, "ball_query_wrapper_cpp");
}
