#include <torch/extension.h>

//把 CUDA kernel 包装成 PyTorch 能调用的函数
torch::Tensor farthest_point_sampling_cuda(torch::Tensor xyz, int npoint, torch::Tensor start_idx);
torch::Tensor ball_query_cuda(torch::Tensor xyz, torch::Tensor new_xyz, double radius, int nsample);

torch::Tensor farthest_point_sampling(torch::Tensor xyz, int64_t npoint, torch::Tensor start_idx) {
    TORCH_CHECK(xyz.is_cuda(), "xyz must be a CUDA tensor");
    TORCH_CHECK(start_idx.is_cuda(), "start_idx must be a CUDA tensor");
    TORCH_CHECK(xyz.is_contiguous(), "xyz must be contiguous");
    TORCH_CHECK(start_idx.is_contiguous(), "start_idx must be contiguous");
    TORCH_CHECK(xyz.scalar_type() == at::kFloat, "xyz must be float32");
    TORCH_CHECK(start_idx.scalar_type() == at::kLong, "start_idx must be int64");
    TORCH_CHECK(xyz.dim() == 3 && xyz.size(2) == 3, "xyz must have shape [B, N, 3]");
    TORCH_CHECK(start_idx.dim() == 1 && start_idx.size(0) == xyz.size(0), "start_idx must have shape [B]");

    return farthest_point_sampling_cuda(xyz, static_cast<int>(npoint), start_idx);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("farthest_point_sampling", &farthest_point_sampling, "Optimized farthest point sampling (CUDA)");
    m.def("ball_query", &ball_query_cuda, "Optimized ball query (CUDA)");
}
