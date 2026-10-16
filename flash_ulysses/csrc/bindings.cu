#include "kittens.cuh"
#include "pyutils/torchutils.cuh"
#include "ops.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("launch", &flash_ulysses::launch);
}
