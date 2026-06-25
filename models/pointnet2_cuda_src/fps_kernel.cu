#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

template <unsigned int block_size>
__global__ void farthest_point_sampling_kernel(
    int b,
    int n,
    int m,
    const float *__restrict__ xyz,
    const int64_t *__restrict__ start_idx,
    float *__restrict__ temp,
    int64_t *__restrict__ idxs) {
    if (m <= 0) {
        return;
    }

    const int batch_index = blockIdx.x;
    xyz += batch_index * n * 3;
    temp += batch_index * n;
    idxs += batch_index * m;

    __shared__ float best_dist[block_size];
    __shared__ int best_idx[block_size];

    int old = static_cast<int>(start_idx[batch_index]);
    idxs[0] = old;

    for (int j = 1; j < m; ++j) {
        const float x1 = xyz[old * 3 + 0];
        const float y1 = xyz[old * 3 + 1];
        const float z1 = xyz[old * 3 + 2];

        float thread_best = -1.0f;
        int thread_best_idx = 0;

        for (int k = threadIdx.x; k < n; k += block_size) {
            const float x2 = xyz[k * 3 + 0];
            const float y2 = xyz[k * 3 + 1];
            const float z2 = xyz[k * 3 + 2];
            const float dx = x2 - x1;
            const float dy = y2 - y1;
            const float dz = z2 - z1;
            const float dist = dx * dx + dy * dy + dz * dz;
            const float min_dist = dist < temp[k] ? dist : temp[k];
            temp[k] = min_dist;

            if (min_dist > thread_best) {
                thread_best = min_dist;
                thread_best_idx = k;
            }
        }

        best_dist[threadIdx.x] = thread_best;
        best_idx[threadIdx.x] = thread_best_idx;
        __syncthreads();

        for (unsigned int offset = block_size / 2; offset > 0; offset >>= 1) {
            if (threadIdx.x < offset && best_dist[threadIdx.x] < best_dist[threadIdx.x + offset]) {
                best_dist[threadIdx.x] = best_dist[threadIdx.x + offset];
                best_idx[threadIdx.x] = best_idx[threadIdx.x + offset];
            }
            __syncthreads();
        }

        old = best_idx[0];
        if (threadIdx.x == 0) {
            idxs[j] = old;
        }
        __syncthreads();
    }
}

static int opt_n_threads(int n) {
    int threads = 1;
    while (threads < n && threads < 1024) {
        threads <<= 1;
    }
    return threads;
}

torch::Tensor farthest_point_sampling_cuda(torch::Tensor xyz, int npoint, torch::Tensor start_idx) {
    const c10::cuda::CUDAGuard device_guard(xyz.device());

    const int b = xyz.size(0);
    const int n = xyz.size(1);

    auto idxs = torch::empty({b, npoint}, start_idx.options());
    auto temp = torch::full({b, n}, 1e10, xyz.options());

    const int threads = opt_n_threads(n);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    switch (threads) {
        case 1024:
            farthest_point_sampling_kernel<1024><<<b, 1024, 0, stream>>>(
                b, n, npoint, xyz.data_ptr<float>(), start_idx.data_ptr<int64_t>(),
                temp.data_ptr<float>(), idxs.data_ptr<int64_t>());
            break;
        case 512:
            farthest_point_sampling_kernel<512><<<b, 512, 0, stream>>>(
                b, n, npoint, xyz.data_ptr<float>(), start_idx.data_ptr<int64_t>(),
                temp.data_ptr<float>(), idxs.data_ptr<int64_t>());
            break;
        case 256:
            farthest_point_sampling_kernel<256><<<b, 256, 0, stream>>>(
                b, n, npoint, xyz.data_ptr<float>(), start_idx.data_ptr<int64_t>(),
                temp.data_ptr<float>(), idxs.data_ptr<int64_t>());
            break;
        case 128:
            farthest_point_sampling_kernel<128><<<b, 128, 0, stream>>>(
                b, n, npoint, xyz.data_ptr<float>(), start_idx.data_ptr<int64_t>(),
                temp.data_ptr<float>(), idxs.data_ptr<int64_t>());
            break;
        case 64:
            farthest_point_sampling_kernel<64><<<b, 64, 0, stream>>>(
                b, n, npoint, xyz.data_ptr<float>(), start_idx.data_ptr<int64_t>(),
                temp.data_ptr<float>(), idxs.data_ptr<int64_t>());
            break;
        case 32:
            farthest_point_sampling_kernel<32><<<b, 32, 0, stream>>>(
                b, n, npoint, xyz.data_ptr<float>(), start_idx.data_ptr<int64_t>(),
                temp.data_ptr<float>(), idxs.data_ptr<int64_t>());
            break;
        case 16:
            farthest_point_sampling_kernel<16><<<b, 16, 0, stream>>>(
                b, n, npoint, xyz.data_ptr<float>(), start_idx.data_ptr<int64_t>(),
                temp.data_ptr<float>(), idxs.data_ptr<int64_t>());
            break;
        case 8:
            farthest_point_sampling_kernel<8><<<b, 8, 0, stream>>>(
                b, n, npoint, xyz.data_ptr<float>(), start_idx.data_ptr<int64_t>(),
                temp.data_ptr<float>(), idxs.data_ptr<int64_t>());
            break;
        case 4:
            farthest_point_sampling_kernel<4><<<b, 4, 0, stream>>>(
                b, n, npoint, xyz.data_ptr<float>(), start_idx.data_ptr<int64_t>(),
                temp.data_ptr<float>(), idxs.data_ptr<int64_t>());
            break;
        case 2:
            farthest_point_sampling_kernel<2><<<b, 2, 0, stream>>>(
                b, n, npoint, xyz.data_ptr<float>(), start_idx.data_ptr<int64_t>(),
                temp.data_ptr<float>(), idxs.data_ptr<int64_t>());
            break;
        default:
            farthest_point_sampling_kernel<1><<<b, 1, 0, stream>>>(
                b, n, npoint, xyz.data_ptr<float>(), start_idx.data_ptr<int64_t>(),
                temp.data_ptr<float>(), idxs.data_ptr<int64_t>());
            break;
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return idxs;
}

__global__ void ball_query_kernel(
    int b,
    int n,
    int s,
    float radius2,
    int nsample,
    const float *__restrict__ xyz,
    const float *__restrict__ new_xyz,
    int64_t *__restrict__ idx) {
    const int query_index = blockIdx.x;
    const int batch_index = blockIdx.y;

    if (query_index >= s || batch_index >= b || threadIdx.x != 0) {
        return;
    }

    xyz += batch_index * n * 3;
    new_xyz += (batch_index * s + query_index) * 3;
    idx += (batch_index * s + query_index) * nsample;

    const float x1 = new_xyz[0];
    const float y1 = new_xyz[1];
    const float z1 = new_xyz[2];

    int count = 0;
    int first_idx = 0;

    for (int k = 0; k < n; ++k) {
        const float x2 = xyz[k * 3 + 0];
        const float y2 = xyz[k * 3 + 1];
        const float z2 = xyz[k * 3 + 2];
        const float dx = x2 - x1;
        const float dy = y2 - y1;
        const float dz = z2 - z1;
        const float dist2 = dx * dx + dy * dy + dz * dz;

        if (dist2 <= radius2) {
            if (count == 0) {
                first_idx = k;
                for (int j = 0; j < nsample; ++j) {
                    idx[j] = first_idx;
                }
            }

            idx[count] = k;
            ++count;

            if (count >= nsample) {
                break;
            }
        }
    }

    if (count == 0) {
        for (int j = 0; j < nsample; ++j) {
            idx[j] = 0;
        }
    }
}

torch::Tensor ball_query_cuda(torch::Tensor xyz, torch::Tensor new_xyz, double radius, int nsample) {
    const c10::cuda::CUDAGuard device_guard(xyz.device());

    TORCH_CHECK(xyz.is_cuda(), "xyz must be a CUDA tensor");
    TORCH_CHECK(new_xyz.is_cuda(), "new_xyz must be a CUDA tensor");
    TORCH_CHECK(xyz.is_contiguous(), "xyz must be contiguous");
    TORCH_CHECK(new_xyz.is_contiguous(), "new_xyz must be contiguous");
    TORCH_CHECK(xyz.scalar_type() == at::kFloat, "xyz must be float32");
    TORCH_CHECK(new_xyz.scalar_type() == at::kFloat, "new_xyz must be float32");
    TORCH_CHECK(xyz.dim() == 3 && xyz.size(2) == 3, "xyz must have shape [B, N, 3]");
    TORCH_CHECK(new_xyz.dim() == 3 && new_xyz.size(2) == 3, "new_xyz must have shape [B, S, 3]");
    TORCH_CHECK(xyz.size(0) == new_xyz.size(0), "xyz and new_xyz must have same batch size");

    const int b = xyz.size(0);
    const int n = xyz.size(1);
    const int s = new_xyz.size(1);

    auto idx = torch::empty({b, s, nsample}, xyz.options().dtype(at::kLong));
    const dim3 blocks(s, b);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    ball_query_kernel<<<blocks, 1, 0, stream>>>(
        b, n, s, static_cast<float>(radius * radius), nsample,
        xyz.data_ptr<float>(), new_xyz.data_ptr<float>(), idx.data_ptr<int64_t>());

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return idx;
}
