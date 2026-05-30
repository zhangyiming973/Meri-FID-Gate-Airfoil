# Meridian SDF 尺寸引导潜空间扩散

基于 `sdf2d` 子午面数据的**两阶段生成管线**：自编码器学习潜表示，再在潜空间上训练**尺寸条件扩散模型**。

## 架构概览

| 阶段      | 说明                                 |
| ------- | ---------------------------------- |
| 1. 自编码器 | SDF + 语义 mask → 潜向量 `z_m` → 重建 SDF |
| 2. 质量门禁 | 校验重建质量与潜空间分布                       |
| 3. 条件扩散 | 以 5 维设计参数为条件生成 `z_m`，再解码为 SDF      |

**条件向量**：`hub_r_end_mm`、`rim_r_start_mm`、`angle_web_deg`、`r_trans_bore_web_mm`、`z_min`

## 方案与目录

每个扩散方案代码**完全独立**，位于 `schemes/` 下，配置放在各自 `config/` 目录：

| 方案       | `--scheme` | 目录                  | 去噪器                  |
| -------- | ---------- | ------------------- | -------------------- |
| MLP 扩散   | `mlp`      | `schemes/mlp/`      | PCA(128) + MLP       |
| PCA-UNet | `pca_unet` | `schemes/pca_unet/` | PCA(128) → 网格 + UNet |

## 数据集与固定划分

每个数据集在 `data/{dataset}/` 下维护元数据与**固定训练/测试划分**：

```
data/
├── single/
│   ├── dataset.json              # 数据集元数据、划分参数
│   └── processed/
│       ├── meridian_index.csv
│       ├── train_split.csv       # 固定训练集（prepare-splits 生成）
│       ├── test_split.csv        # 固定测试集
│       ├── condition_stats.json
│       └── meridian_samples/
└── F404/
    ├── dataset.json
    └── processed/
        ├── train_split.csv
        ├── test_split.csv
        └── ...
```

训练任务直接从 `data/{dataset}/processed/train_split.csv` 和 `test_split.csv` 读取，不再每次随机划分。

## 统一入口 `run.py`

所有数据准备、训练、测试均通过一条命令启动，用 `--scheme` 和 `--dataset` 区分方案与数据：

```bash
# 0. 首次使用：生成固定划分（每个数据集执行一次）
python run.py prepare-splits --dataset single
python run.py prepare-splits --dataset F404

# 1. 训练 — MLP 方案 + F404 数据集（全流程）
python run.py train --scheme mlp --dataset F404

# 2. 训练 — PCA-UNet 方案 + single 数据集
python run.py train --scheme pca_unet --dataset single

# 3. 分阶段训练
python run.py train --scheme mlp --dataset F404 --stage ae
python run.py train --scheme mlp --dataset F404 --stage diff \
  --ae-run-dir outputs/mlp/F404/{timestamp}/autoencoder

python run.py train --scheme pca_unet --dataset F404 --stage unet \
  --ae-run-dir outputs/pca_unet/F404/{timestamp}/autoencoder

# 4. 快速冒烟
python run.py train --scheme mlp --dataset F404 --fast

# 5. 测试 / 可视化
python run.py test --scheme mlp --dataset F404 \
  --run-dir outputs/mlp/F404/{timestamp}
```

配置文件自动从 `schemes/{scheme}/config/{dataset}.json` 读取，无需手动指定 `--config`。

## 输出目录

统一格式：`outputs/{方案}/{数据集}/{时间戳}/`

```
outputs/mlp/F404/20260529_234932/
├── logs/train.log
├── autoencoder/          # AE 权重、潜向量、Gate 报告、重建可视化
└── diffusion/            # 扩散模型、生成指标、生成可视化

outputs/pca_unet/single/20260529_120000/
├── logs/train.log
├── autoencoder/
└── diffusion/
```

## 项目结构

```
meri-fid-gate/
├── run.py                      # 统一入口
├── scripts/
│   ├── prepare_splits.py       # 生成固定 train/test 划分
│   ├── split_utils.py
│   └── build_schemes.py        # 从 src 重建 scheme 包（开发用）
├── data/
│   ├── single/dataset.json
│   └── F404/dataset.json
├── schemes/
│   ├── mlp/                    # MLP 方案（独立代码）
│   │   ├── config/single.json
│   │   ├── config/F404.json
│   │   ├── pipeline.py
│   │   ├── models/
│   │   ├── train/
│   │   └── ...
│   └── pca_unet/               # PCA-UNet 方案（独立代码）
│       ├── config/
│       ├── pipeline.py
│       └── ...
└── outputs/{scheme}/{dataset}/{timestamp}/
```

## 环境安装

```bash
pip install -r requirements.txt
```

## F404 数据注意事项

F404 需在 `data/F404/dataset.json` 中设置：

- `split.filter_passed_quality: false`（当前全部 `passed_quality=False`）
- `condition_specs_file`（自动从 JSONL 补齐 5 维条件列）

## 添加新方案

1. 在 `schemes/` 下新建目录（如 `schemes/my_scheme/`），复制并独立维护全套代码
2. 在 `schemes/my_scheme/config/` 下为每个数据集添加 JSON 配置
3. 在 `run.py` 的 `SCHEMES` 字典中注册方案名与 `pipeline` 模块

## 技术架构详解

### 1. 自编码器（MeridianAutoEncoder）

**架构设计**：

- **Encoder**：4层 ConvBlock + MaxPool2d，逐步压缩空间维度
  - 输入：256×256 SDF（可选拼接语义 mask，共 2 通道）
  - 输出：64×16×16 潜向量 `z_m`
- **Decoder**：4层 ConvBlock + Upsample，逐步恢复空间维度
  - 输入：64×16×16 潜向量
  - 输出：256×256 重建 SDF

**ConvBlock 结构**：

```
Conv2d(3×3) → GroupNorm → SiLU → Conv2d(3×3) → GroupNorm → SiLU
```

**损失函数设计**：

- **L1 重建损失**：全局 SDF 重建精度
- **L2 重建损失**：像素级重建精度
- **零等值面加权损失**：SDF≈0 区域赋予更高权重（高斯衰减，sigma=1.5mm）
- **物理尺寸约束**：最小壁厚与截面面积误差
- **语义一致性损失**：hub/web/rim 三区域覆盖率一致性

### 2. PCA 降维（LatentPCA）

**作用**：

- 将 64×16×16=16384 维潜向量压缩到 128 维
- 降低扩散模型计算复杂度
- 保留主要变化方向（SVD 分解）

**流程**：

```
z_m (64×16×16) → flatten → PCA 投影 → w (128 维) → 归一化
```

### 3. MLP 去噪器（MLPDenoiser）

**架构**：

- **正弦位置编码**：将离散时间步 t 映射为 128 维连续向量
  ```
  emb(t) = [sin(t·ω₁), cos(t·ω₁), ..., sin(t·ω_k), cos(t·ω_k)]
  ω_i = exp(-log(10000)·i/k)
  ```
- **MLP 网络**：4 层全连接网络（512 维隐层）
  ```
  输入拼接：[带噪潜向量 w_t | 工况 condition | 时间嵌入 t_emb]
  → Linear(128+5+128 → 512) → SiLU → Linear(512 → 512) → SiLU → ...
  → 输出：预测噪声 ε，形状与 w_t 相同
  ```

### 4. PCA-UNet 去噪器（ConditionalUNet）

**架构设计**：

- **PCA → 网格化**：128 维向量 reshape 为 1×8×16 网格
- **Encoder-Decoder UNet**：
  - Encoder：3层下采样（Conv + GroupNorm + SiLU）
  - Decoder：3层上采样（Upsample + Conv + GroupNorm + SiLU）
  - Skip Connection：保留空间细节
- **条件注入**：工况向量广播到空间维度后与噪声 latent 拼接

### 5. 质量门控（LatentRepresentationGate）

**评估指标**：

- **重建 L1 误差**：每样本 L1 误差 ≤ `max_recon_l1`（默认 0.08）
- **潜空间标准差**：全局潜变量 std ≥ `min_latent_std`（默认 0.05）
- **通过率**：合格样本比例 ≥ `min_pass_ratio`（默认 85%）

**作用**：

- 过滤重建质量差的样本
- 确保潜空间分布合理（避免坍缩）
- 决定是否进入扩散训练阶段

### 6. 扩散训练（DDPM/DDIM）

**关键技术**：

- **Classifier-Free Guidance (CFG)**：
  - 训练时：随机 dropout 条件向量（概率 20%）
  - 推理时：混合条件与无条件预测
    ```
    ε_pred = ε_uncond + cfg_scale·(ε_cond - ε_uncond)
    ```
- **DDIM 采样**：加速推理（50 步而非 200 步）
- **物理引导（可选）**：
  - 在采样过程中加入应力约束
  - 权重：0.02，sigma\_vm\_limit：103000

## 数据流详解

### 输入数据格式

**NPZ 文件内容**：

- `sdf`：256×256 符号距离场（负值表示实体内部）
- `semantic`：语义 mask（hub/web/rim 区域标记）
- `condition_json`：5 维工况向量（可选）
- `physics`：物理属性（最小壁厚、截面面积）

**条件向量来源**：

1. `meridian_index.csv`：直接包含 5 维工况列
2. `param_table.csv`：参数表补全缺失列
3. `condition_specs.jsonl`：从几何规格推导工况

### 训练/测试划分

**固定划分机制**：

- `prepare-splits` 命令生成固定划分 CSV
- 确保每次训练使用相同数据集
- 支持质量过滤（`passed_quality` 字段）

### 数据预处理

**条件归一化**：

- 训练集统计均值/标准差 → `condition_stats.json`
- 归一化：`(value - mean) / std`
- 推理时反归一化恢复原始尺度

## 关键技术点

### 1. 正弦位置编码（SinusoidalPosEmb）

**原理**：

- 将离散时间步 t ∈ \[0, T] 映射为连续向量
- 不同频率捕捉不同时间尺度特征
- Transformer 经典做法，适用于扩散模型

### 2. Classifier-Free Guidance (CFG)

**训练阶段**：

- 随机 dropout 条件向量（概率 cfg\_dropout=0.2）
- 学习条件与无条件两种模式

**推理阶段**：

- 同时计算条件与无条件预测
- 按 cfg\_scale（默认 1.5）加权混合
- 提升生成质量与条件控制强度

### 3. DDIM 采样加速

**原理**：

- 非马尔可夫采样过程
- 从 200 步压缩到 50 步
- 保持生成质量

### 4. 潜空间扩散优势

**相比像素空间扩散**：

- 计算效率高：128 维而非 256×256
- 学习紧凑表示：去除冗余信息
- 更好泛化：PCA 保留主要变化方向

## 模型参数说明

### 自编码器参数

| 参数                | 默认值    | 说明                         |
| ----------------- | ------ | -------------------------- |
| `in_channels`     | 2      | 输入通道数（SDF + semantic mask） |
| `latent_channels` | 64     | 潜向量通道数                     |
| `latent_spatial`  | 16     | 潜向量空间尺寸（16×16）             |
| `epochs`          | 80     | 训练轮数                       |
| `batch_size`      | 8      | 批大小                        |
| `lr`              | 0.0003 | 学习率                        |

### 扩散模型参数

| 参数             | 默认值    | 说明                |
| -------------- | ------ | ----------------- |
| `pca_dim`      | 128    | PCA 压缩维度          |
| `mlp_hidden`   | 512    | MLP 隐层宽度          |
| `timesteps`    | 200    | 扩散时间步数            |
| `epochs`       | 200    | 训练轮数              |
| `batch_size`   | 16     | 批大小               |
| `lr`           | 0.0005 | 学习率               |
| `cfg_dropout`  | 0.2    | CFG 条件 dropout 概率 |
| `cfg_scale`    | 1.5    | CFG 推理强度          |
| `sample_steps` | 50     | DDIM 采样步数         |
| `use_ddim`     | true   | 使用 DDIM 采样        |

### 质量门控参数

| 参数               | 默认值  | 说明           |
| ---------------- | ---- | ------------ |
| `max_recon_l1`   | 0.08 | 最大重建 L1 误差阈值 |
| `min_latent_std` | 0.05 | 最小潜空间标准差阈值   |
| `min_pass_ratio` | 0.85 | 最小样本通过率阈值    |

### UNet 参数（PCA-UNet 方案）

| 参数             | 默认值 | 说明         |
| -------------- | --- | ---------- |
| `unet_base_ch` | 64  | UNet 基础通道数 |
| `time_dim`     | 128 | 时间嵌入维度     |

## 参考指标（single 数据集）

| 方案       | mean\_gen\_l1 | AE 重建 L1 |
| -------- | ------------- | -------- |
| MLP      | \~0.036       | \~0.008  |
| PCA-UNet | \~0.014       | \~0.008  |

#
