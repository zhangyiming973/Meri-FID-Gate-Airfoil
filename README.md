# Meridian SDF 尺寸引导潜空间扩散

基于 `sdf2d` 子午面数据的**两阶段生成管线**：先用自编码器学习紧凑潜表示，再在潜空间上训练**尺寸条件扩散模型**，根据关键几何参数生成 SDF 形状。

## 功能概览

| 阶段 | 模块 | 说明 |
|------|------|------|
| 1 | `MeridianAutoEncoder` | 输入 SDF（可选拼接语义 mask）→ 潜向量 `z_m` → 重建 SDF |
| 2 | `LatentRepresentationGate` | 校验重建质量与潜空间分布，决定是否进入扩散训练 |
| 3 | 条件扩散 | 在压缩潜空间上，以尺寸参数为条件生成 `z_m`，再经 AE 解码为 SDF |

**条件向量（5 维）**：`hub_r_end_mm`、`rim_r_start_mm`、`angle_web_deg`、`r_trans_bore_web_mm`、`z_min`

## 扩散方案对比

| 方案 | 入口 | 去噪器 | 输出目录 |
|------|------|--------|----------|
| **MLP** | `train.py` | PCA(128) + MLP | `outputs/{ds}/pipeline/` |
| **PCA-UNet** | `train_unet.py` | PCA(128) → 网格 + UNet | `outputs/{ds}/unet_pipeline/` |

### PCA-UNet 独立入口

```bash
# 全流程：AE + UNet 扩散
PYTHONPATH=. python3 train_unet.py

# 仅训练 UNet（复用已有 AE）
PYTHONPATH=. python3 train_unet.py --stage unet \
  --ae-run-dir outputs/single/unet_pipeline/XXXX/autoencoder

# 快速冒烟
PYTHONPATH=. python3 train_unet.py --fast

# 可视化
PYTHONPATH=. python3 visualize_unet.py \
  --ae-run-dir outputs/single/unet_pipeline/XXXX/autoencoder \
  --unet-run-dir outputs/single/unet_pipeline/XXXX/diffusion_unet
```

配置：`configs/train_unet.json`（`unet` 配置块，与 `train.json` 独立）

### PCA-UNet 架构

```
z_m (64×16×16)
    ↓ PCA 压缩
w (128 维向量)
    ↓ reshape
grid (1×8×16)
    ↓ Conditional UNet 扩散 (DDIM + CFG)
grid̂
    ↓ flatten + PCA 逆变换
ẑ_m → AE Decoder → SDF
```

UNet 采用 **encoder-decoder + skip connection**，条件向量广播到空间维度后与噪声 latent 拼接。

## 项目结构

```
project/
├── train.py                # MLP 扩散管线
├── train_unet.py           # PCA-UNet 独立管线
├── visualize.py
├── visualize_unet.py
├── configs/
│   ├── train.json          # MLP 配置
│   └── train_unet.json     # UNet 配置（unet 块）
├── src/train/
│   ├── train_diffusion.py      # MLP 训练
│   └── train_unet_diffusion.py # UNet 训练
└── outputs/{dataset}/pipeline/{timestamp}/
```

## 环境安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 全流程（MLP 方案）

```bash
PYTHONPATH=. python3 train.py --config configs/train.json
```

## 配置说明（`train_unet.json` → `unet` 块）

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `pca_dim` | 128 | PCA 压缩维度 |
| `unet_base_ch` | 64 | UNet 基础通道数 |
| `cfg_scale` | 1.5 | Classifier-Free Guidance 强度 |
| `use_ddim` | true | DDIM 采样 |

## 参考指标

| 方案 | mean_gen_l1 | AE 重建 L1 |
|------|-------------|------------|
| MLP (PCA) | ~0.036 | ~0.008 |
| 原始 UNet | ~0.76 | ~0.008 |

## 许可证

内部研究项目，按需补充许可证信息。
