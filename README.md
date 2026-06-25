# PointNet++ S3DIS Semantic Segmentation with Dual Attention

本项目基于 PyTorch 版 PointNet / PointNet++，当前主要面向 S3DIS 室内点云语义分割任务。代码在不改变 PointNet++ 主干采样与聚合流程的前提下，对 Set Abstraction 模块加入轻量级双注意力机制：

- SE Channel Attention：对通道维度进行自适应重标定。
- Spatial Attention：对采样后的点维度生成逐点权重。
- Residual Fusion：保留原始 SA 输出，并与注意力增强特征残差融合。

FPS、ball query、grouping、Feature Propagation、数据加载、损失函数和测试流程均保持原有结构。

另外，FPS 和 ball query 已增加可选 CUDA extension 加速路径：Linux + CUDA + nvcc 环境下会自动 JIT 编译并使用优化版采样/邻域查询；如果环境不满足要求，会自动回退到原 PyTorch 实现。

## 当前模型结构

语义分割入口模型为：

```text
models/pointnet2_sem_seg.py
```

整体网络结构：

```text
Input point cloud [B, 9, N]
        |
        | xyz = first 3 channels
        | points = all 9 channels
        v
SA1: PointNetSetAbstraction
        |
SA2: PointNetSetAbstraction
        |
SA3: PointNetSetAbstraction
        |
SA4: PointNetSetAbstraction
        |
FP4: PointNetFeaturePropagation
        |
FP3: PointNetFeaturePropagation
        |
FP2: PointNetFeaturePropagation
        |
FP1: PointNetFeaturePropagation
        |
Conv1d + BN + ReLU + Dropout
        |
Conv1d classifier
        |
LogSoftmax
        |
Output [B, N, num_classes]
```

S3DIS 默认类别数为 13。

## SA 模块改动

核心改动位于：

```text
models/pointnet2_utils.py
```

修改类：

```python
class PointNetSetAbstraction(nn.Module)
```

原始 SA 流程保持不变：

```text
FPS sampling
    |
Ball query grouping
    |
Shared MLP
    |
Max pooling over local neighborhood
```

在 MLP + max pooling 得到 `[B, C, S]` 特征后，新增：

```text
SA output [B, C, S]
    |
SE Channel Attention
    |
Spatial Attention
    |
Residual Fusion
    |
Enhanced SA output [B, C, S]
```

注意力计算：

```text
identity = new_points

SE:
    global average pooling over point dimension
    Linear(C, C / 4)
    ReLU
    Linear(C / 4, C)
    Sigmoid
    channel-wise reweight

Spatial:
    Conv1d(C, 1, kernel_size=1)
    Sigmoid
    point-wise reweight

Residual:
    new_points = identity + attention_features
```

代码中已标注：

```python
# added SE attention
# added spatial attention
# residual fusion
```

## 项目目录

```text
.
├── data/
│   └── stanford_indoor3d/          # S3DIS 预处理后的 .npy 数据
├── data_utils/
│   ├── S3DISDataLoader.py          # S3DIS 数据加载
│   ├── collect_indoor3d_data.py    # S3DIS 原始数据预处理
│   └── indoor3d_util.py
├── models/
│   ├── pointnet2_sem_seg.py        # PointNet++ SSG 语义分割网络
│   ├── pointnet2_sem_seg_msg.py    # PointNet++ MSG 语义分割网络
│   ├── pointnet2_utils.py          # SA / FP / FPS / grouping 实现
│   ├── pointnet2_cuda_ops.py       # 可选 CUDA FPS / ball query 加速入口
│   ├── pointnet2_cuda_src/         # CUDA extension 源码
│   ├── pointnet_sem_seg.py         # PointNet 语义分割网络
│   └── ...
├── log/
│   └── sem_seg/                    # 训练日志、权重、可视化结果
├── visualizer/                     # 点云可视化工具
├── train_semseg.py                 # S3DIS 语义分割训练入口
├── test_semseg.py                  # S3DIS 语义分割测试入口
├── provider.py                     # 数据增强工具
└── README.md
```

## 环境要求

推荐使用带 CUDA 的 PyTorch 环境。训练脚本中模型和 loss 默认调用 `.cuda()`，因此需要 GPU 版 PyTorch。

如果要启用优化版 FPS，需要 Linux 环境中具备：

```text
PyTorch CUDA build
CUDA Toolkit / nvcc
可用 GPU
可用 C++ 编译器
```

检查 PyTorch：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

如果输出 `True`，说明 CUDA 可用。

检查 `nvcc`：

```bash
nvcc --version
```

第一次训练时，PyTorch 会自动编译 `models/pointnet2_cuda_src/` 下的 CUDA extension。编译成功后会缓存到 PyTorch extensions 目录，后续运行会复用缓存。

## 数据准备

原始 S3DIS 数据放置路径：

```text
data/s3dis/Stanford3dDataset_v1.2_Aligned_Version/
```

执行预处理：

```bash
cd data_utils
python collect_indoor3d_data.py
cd ..
```

预处理完成后数据应位于：

```text
data/stanford_indoor3d/
```

当前仓库中已经存在 `data/stanford_indoor3d/` 时，可以直接训练。

## 训练

在项目根目录运行：

```bash
python train_semseg.py --model pointnet2_sem_seg --test_area 5 --log_dir pointnet2_sem_seg_dual_attn --batch_size 16 --epoch 32
```

参数说明：

```text
--model       使用的模型文件名，对应 models/pointnet2_sem_seg.py
--test_area   S3DIS 留作测试的 Area，常用 5
--log_dir     日志和 checkpoint 保存目录
--batch_size  训练 batch size
--epoch       训练轮数
```

如果显存不足，可以降低 batch size：

```bash
python train_semseg.py --model pointnet2_sem_seg --test_area 5 --log_dir pointnet2_sem_seg_dual_attn --batch_size 8 --epoch 32
```

## 基于预训练权重训练

如果要从已有 PointNet++ 语义分割 checkpoint 初始化当前双注意力模型，使用 `--pretrained_checkpoint` 指向原始权重，并使用一个新的 `--log_dir` 保存本次实验。

示例：

```bash
python train_semseg.py \
  --model pointnet2_sem_seg \
  --test_area 5 \
  --log_dir pointnet2_sem_seg_dual_attn_pretrained \
  --pretrained_checkpoint log/sem_seg/pointnet2_sem_seg/checkpoints/best_model.pth \
  --batch_size 16 \
  --epoch 32
```

该模式下：

```text
读取原始 checkpoint 作为初始化
不覆盖 --pretrained_checkpoint 指向的文件
从 epoch 0 开始训练
新权重保存到新的 log_dir/checkpoints/
新增注意力层参数会随机初始化并参与训练
```

如果需要严格冻结从预训练 checkpoint 加载到的参数，只训练新增/未加载的参数，例如新增注意力层：

```bash
python train_semseg.py \
  --model pointnet2_sem_seg \
  --test_area 5 \
  --log_dir pointnet2_sem_seg_dual_attn_freeze_pretrained \
  --pretrained_checkpoint log/sem_seg/pointnet2_sem_seg/checkpoints/best_model.pth \
  --freeze_pretrained \
  --batch_size 16 \
  --epoch 32
```

训练输出保存到：

```text
log/sem_seg/pointnet2_sem_seg_dual_attn/
├── checkpoints/
│   └── best_model.pth
├── logs/
└── pointnet2_sem_seg.py
```

训练时脚本会复制当前模型文件和 `pointnet2_utils.py` 到日志目录，便于复现实验。

## 测试

训练完成后运行：

```bash
python test_semseg.py --log_dir pointnet2_sem_seg_dual_attn --test_area 5 --visual
```

测试脚本会读取：

```text
log/sem_seg/pointnet2_sem_seg_dual_attn/checkpoints/best_model.pth
```

如果使用 `--visual`，可视化结果会保存到：

```text
log/sem_seg/pointnet2_sem_seg_dual_attn/visual/
```

`.obj` 文件可以使用 MeshLab 打开查看。

## 关键代码位置

语义分割网络：

```text
models/pointnet2_sem_seg.py
```

SA / FP / FPS / grouping：

```text
models/pointnet2_utils.py
```

CUDA FPS / ball query 加速：

```text
models/pointnet2_cuda_ops.py
models/pointnet2_cuda_src/fps.cpp
models/pointnet2_cuda_src/fps_kernel.cu
```

S3DIS 数据加载：

```text
data_utils/S3DISDataLoader.py
```

训练入口：

```text
train_semseg.py
```

预训练参数：

```text
--pretrained_checkpoint  从指定 checkpoint 初始化，不覆盖源文件
--freeze_pretrained      冻结已加载的预训练参数，只训练新增/未加载参数
```

测试入口：

```text
test_semseg.py
```

## 当前改动范围

已修改：

```text
models/pointnet2_utils.py
models/pointnet2_cuda_ops.py
models/pointnet2_cuda_src/
train_semseg.py
README.md
```

未修改：

```text
data_utils/
test_semseg.py
loss function
evaluation pipeline
grouping / sampling interface
Feature Propagation logic
```

## 注意事项

- 当前双注意力只加入到 `PointNetSetAbstraction`，即 SSG SA 模块。
- `PointNetSetAbstractionMsg` 未改动，因此 `pointnet2_sem_seg_msg.py` 不会使用这次新增的双注意力。
- 如果要训练当前改进模型，请使用 `--model pointnet2_sem_seg`。
- 如果加载旧 checkpoint，可能会因为新增注意力层参数而出现权重不匹配。建议使用新的 `--log_dir` 从头训练。
- 优化版 FPS / ball query 只在 CUDA tensor、float32、shape 为 `[B, N, 3]` 时启用；否则自动回退。
- 如需强制关闭 CUDA extension，可设置环境变量：`POINTNET2_DISABLE_CUDA_OPS=1`。
- 如需查看 CUDA extension 编译日志，可设置环境变量：`POINTNET2_OPS_VERBOSE=1`。

## Reference

- PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
- PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space
- Original PyTorch implementation: `yanx27/Pointnet_Pointnet2_pytorch`
