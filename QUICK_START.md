# 快速开始指南 (Quick Start Guide)

## 训练深度融合模型

### 步骤1: 准备环境

```bash
# 确保已安装所有依赖
pip install torch tensorboardX opencv-python pillow numpy

# 确保检查点文件存在
ls -la checkpoints/model_000014.ckpt
```

### 步骤2: 修改配置文件

编辑 `train_fusion.sh`，设置数据路径：

```bash
# 修改这些路径为你的实际路径
TRAINPATH="/data1/local_userdata/houbosen/dtu_training_raw"
TESTPATH="/data1/local_userdata/houbosen/dtu_training_raw"
TRAINLIST="lists/dtu/train.txt"
TESTLIST="lists/dtu/val.txt"
```

### 步骤3: 启动训练

```bash
# 方法1: 使用启动脚本
bash train_fusion.sh

# 方法2: 直接使用Python
python train_fusion.py \
    --trainpath /your/data/path \
    --trainlist lists/dtu/train.txt \
    --testpath /your/data/path \
    --testlist lists/dtu/val.txt \
    --mvsnet_ckpt checkpoints/model_000014.ckpt \
    --logdir checkpoints/fusion \
    --epochs 20 \
    --batch_size 4 \
    --lr 0.0001
```

### 步骤4: 监控训练过程

在另一个终端打开TensorBoard：

```bash
tensorboard --logdir checkpoints/fusion
```

然后在浏览器中打开 `http://localhost:6006`

### 步骤5: 恢复中断的训练

如果训练中断，可以从最后一个检查点恢复：

```bash
python train_fusion.py \
    --trainpath /your/data/path \
    --trainlist lists/dtu/train.txt \
    --mvsnet_ckpt checkpoints/model_000014.ckpt \
    --logdir checkpoints/fusion \
    --resume
```

---

## 使用训练好的模型进行推理

### 步骤1: 准备模型

```bash
# 查看可用的检查点
ls -la checkpoints/fusion/model_*.ckpt
```

### 步骤2: 运行推理

```bash
python infer_fusion.py \
    --testpath /your/data/path \
    --testlist lists/dtu/val.txt \
    --mvsnet_ckpt checkpoints/model_000014.ckpt \
    --fusion_ckpt checkpoints/fusion/model_000019.ckpt \
    --outdir ./outputs_fusion \
    --save_depth \
    --save_conf \
    --display
```

### 步骤3: 查看输出

输出文件将保存在 `outputs_fusion/` 目录：

```bash
ls outputs_fusion/
# depth_fused_000000.pfm
# depth_mvs_000000.pfm
# conf_fused_000000.npy
# conf_mvs_000000.npy
# depth_fused_000000.png  (可视化)
```

---

## 主要参数调优指南

### 如果模型欠拟合 (Underfitting)
- 增加训练轮数: `--epochs 30`
- 降低学习率衰减倍数，改变 `--lrepochs` 参数
- 增加隐层维度: `--hidden_dim 128`
- 增加bin数: `--num_bins 128`

### 如果模型过拟合 (Overfitting)
- 增加权重衰减: `--wd 5e-4`
- 增加学习率衰减速度
- 减少隐层维度: `--hidden_dim 32`
- 减少bin数: `--num_bins 32`

### 如果训练速度慢
- 增加批次大小: `--batch_size 8`
- 减少验证频率: `--summary_freq 50`
- 使用多GPU（如果可用）

### 如果显存不足
- 减少批次大小: `--batch_size 2`
- 减少图像分辨率（修改数据集代码）
- 减少 `--numdepth` 的数值

---

## 数据输入格式

### 数据集要求

数据应该组织为以下结构：

```
your_data_path/
├── Rectified/               # 原始多视图图像 (1600x1200)
│   ├── scan001/
│   │   ├── rect_001_0_r5000.png
│   │   ├── rect_002_0_r5000.png
│   │   └── ...
│   └── scan002/
│       └── ...
├── Depths_raw/              # 参考视图深度图 (1184x1600)
│   ├── scan001/
│   │   ├── depth_map_0000.pfm
│   │   ├── depth_visual_0000.png
│   │   └── ...
│   └── ...
├── DA3Depth/               # DA3预测的深度 (可选)
│   ├── scan001/
│   │   ├── 00000000.npy
│   │   └── ...
│   └── ...
├── DA3Conf/                # DA3置信度 (可选)
│   ├── scan001/
│   │   ├── 00000000.npy
│   │   └── ...
│   └── ...
├── Cameras/
│   ├── train/
│   │   ├── 00000000_cam.txt
│   │   └── ...
│   └── pair.txt
└── lists/
    ├── train.txt           # 训练扫描列表 (每行一个scan名)
    ├── val.txt
    └── test.txt
```

### 列表文件格式

`train.txt`, `val.txt`, `test.txt` 中每行包含一个扫描名：

```
scan001
scan002
scan003
...
```

---

## 输出文件说明

### 检查点文件 (Checkpoint)

每个检查点包含：
- `epoch`: 当前训练轮数
- `model`: 融合模型的权重
- `optimizer`: 优化器的状态

加载检查点：
```python
state_dict = torch.load('model_000010.ckpt')
fusion_model.load_state_dict(state_dict['model'])
optimizer.load_state_dict(state_dict['optimizer'])
epoch = state_dict['epoch']
```

### 推理输出

- `.pfm`: 深度图（浮点格式）
- `.npy`: 置信度图（NumPy格式）
- `.png`: 深度可视化（彩色图像）

---

## 常见错误处理

### 错误1: "CUDA out of memory"
```bash
# 解决方案：减少批次大小
python train_fusion.py ... --batch_size 2
```

### 错误2: "FileNotFoundError: lists/dtu/train.txt"
```bash
# 解决方案：确保列表文件存在
ls lists/dtu/
# 如果不存在，手动创建或修改路径
```

### 错误3: "No such file or directory: .../rect_*.png"
```bash
# 解决方案：检查数据路径是否正确
ls your_data_path/Rectified/
```

### 错误4: "RuntimeError: Expected all tensors to be on the same device"
```bash
# 解决方案：确保所有数据都在CUDA上
# 代码中的 tocuda() 函数应该处理这个
```

---

## 性能基准 (Benchmarks)

在典型硬件上的表现（参考值）：

| 配置 | 训练速度 | 内存占用 |
|------|--------|--------|
| batch_size=4, GPU=1x V100 | ~0.5s/sample | ~16GB |
| batch_size=2, GPU=1x V100 | ~0.4s/sample | ~12GB |
| batch_size=8, GPU=4x V100 | ~1.5s/batch | ~20GB (per GPU) |

---

## 进阶用法

### 自定义融合模型

编辑 `fusion_models/depthbin_fusionmodel.py` 来修改网络架构：

```python
class DepthBinFusionNet(nn.Module):
    def __init__(self, num_bins=64, hidden=64):
        super().__init__()
        # 修改这里
        self.encoder = nn.Sequential(
            # 自定义层...
        )
```

### 自定义损失函数

在 `train_fusion.py` 中修改 `train_sample()` 函数：

```python
# 使用自定义损失
loss = your_custom_loss(prob, depth_gt, depth_values)
```

### 添加自定义评估指标

在 `test_sample()` 函数中添加新指标：

```python
scalar_outputs["your_metric"] = compute_your_metric(depth_fused, depth_gt)
```

---

## 获取帮助

如有问题，请检查：

1. **日志输出**: 查看 `checkpoints/fusion/` 中的TensorBoard日志
2. **Python错误堆栈**: 完整的错误信息会显示在终端
3. **数据文件**: 验证所有输入文件是否存在和格式正确
4. **显存占用**: 使用 `nvidia-smi` 监控GPU使用情况

```bash
# 实时监控GPU
nvidia-smi -l 1

# 查看TensorBoard日志
tensorboard --logdir checkpoints/fusion --port 6007
```

---

**祝你训练顺利！** 🚀
