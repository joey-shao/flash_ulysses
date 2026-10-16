#pragma once

#include <ATen/ATen.h>

namespace kittens::py { struct TKParallelTensor; }

namespace flash_ulysses {
void launch(const at::Tensor& x, const at::Tensor& qkv_weight,
            const at::Tensor& positions, const at::Tensor& q_gamma,
            const at::Tensor& k_gamma, const at::Tensor& rope,
            kittens::py::TKParallelTensor& Q, kittens::py::TKParallelTensor& K,
            kittens::py::TKParallelTensor& V, kittens::py::TKParallelTensor& barrier,
            int q_heads, int kv_heads, int sp_size,
            double q_epsilon, double k_epsilon);
}  // namespace flash_ulysses
