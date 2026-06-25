import os
from pathlib import Path

import torch

_FPS_EXT = None
_FPS_LOAD_FAILED = False


def _load_fps_ext():
    global _FPS_EXT, _FPS_LOAD_FAILED

    if _FPS_EXT is not None:
        return _FPS_EXT
    if _FPS_LOAD_FAILED:
        return None
    if os.environ.get("POINTNET2_DISABLE_CUDA_OPS", "0") == "1":
        _FPS_LOAD_FAILED = True
        return None
    if not torch.cuda.is_available():
        _FPS_LOAD_FAILED = True
        return None

    try:
        from torch.utils.cpp_extension import load

        src_dir = Path(__file__).resolve().parent / "pointnet2_cuda_src"
        _FPS_EXT = load(
            name="pointnet2_fps_ext",
            sources=[
                str(src_dir / "fps.cpp"),
                str(src_dir / "fps_kernel.cu"),
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=os.environ.get("POINTNET2_OPS_VERBOSE", "0") == "1",
        )
        return _FPS_EXT
    except Exception as exc:
        _FPS_LOAD_FAILED = True
        print("Warning: failed to load optimized CUDA FPS, using PyTorch fallback.")
        print(exc)
        return None


def farthest_point_sample_cuda(xyz, npoint):
    if not xyz.is_cuda or xyz.dtype != torch.float32 or xyz.shape[-1] != 3:
        return None

    ext = _load_fps_ext()
    if ext is None:
        return None

    xyz = xyz.contiguous()
    start_idx = torch.randint(0, xyz.shape[1], (xyz.shape[0],), dtype=torch.long, device=xyz.device)
    return ext.farthest_point_sampling(xyz, int(npoint), start_idx)


def query_ball_point_cuda(radius, nsample, xyz, new_xyz):
    if (
        not xyz.is_cuda
        or not new_xyz.is_cuda
        or xyz.dtype != torch.float32
        or new_xyz.dtype != torch.float32
        or xyz.shape[-1] != 3
        or new_xyz.shape[-1] != 3
    ):
        return None

    ext = _load_fps_ext()
    if ext is None:
        return None

    return ext.ball_query(xyz.contiguous(), new_xyz.contiguous(), float(radius), int(nsample))
