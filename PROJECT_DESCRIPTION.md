# Meri-FID-Gate-Airfoil 项目详细说明

> 基于条件扩散模型的几何生成系统——从子午面 SDF 到翼型气动校核的完整技术栈  
> 日期：2026-06-15  
> 分支：`airfoil-conditional-diffusion`  
> 根目录：`/home/vipuser/Meri-FID-Gate-Airfoil`

---

## 目录

1. [项目定位与问题定义](#1-项目定位与问题定义)
2. [核心架构：三阶段条件扩散管线](#2-核心架构三阶段条件扩散管线)
3. [六套方案详解](#3-六套方案详解)
4. [统一入口 run.py](#4-统一入口-runpy)
5. [数据管线](#5-数据管线)
6. [网络架构详解](#6-网络架构详解)
7. [损失函数体系](#7-损失函数体系)
8. [扩散模型训练与采样](#8-扩散模型训练与采样)
9. [输出产物与目录规范](#9-输出产物与目录规范)
10. [UIUC 翼型条件扩散实验](#10-uiuc-翼型条件扩散实验)
11. [XFOIL 气动校核子系统](#11-xfoil-气动校核子系统)
12. [代码组织全景图](#12-代码组织全景图)
13. [配置文件参考](#13-配置文件参考)
14. [常用命令速查](#14-常用命令速查)
15. [指标参考](#15-指标参考)
16. [添加新数据集/新方案指南](#16-添加新数据集新方案指南)
17. [Git 提交历史 (airfoil-conditional-diffusion)](#17-git-提交历史-airfoil-conditional-diffusion)

---

## 1. 项目定位与问题定义

### 1.1 任务定义

**原始任务**（子午面）：
- **输入**：5 维设计参数（轮毂半径、轮缘半径、腹板倾角、过渡半径、轴向下界），c ∈ ℝ⁵
- **输出**：256×256 子午面 SDF（Signed Distance Field，负=实体内部），SDF ∈ ℝ²⁵⁶×²⁵⁶

**扩展任务**（翼型）：
- **输入**：5 维几何条件（最大厚度、最大厚度位置、最大弯度、最大弯度位置、尾缘间隙），c ∈ ℝ⁵
- **输出**：128×256 翼型 SDF，进一步提取为上下表面翼型坐标

### 1.2 核心方法

采用**两阶段潜空间扩散生成**框架：

```
阶段 1 (自编码器):  SDF → Encoder → 潜向量 z_m → Decoder → SDF_recon
阶段 2 (质量门控):  校验重建质量与潜空间分布
阶段 3 (条件扩散):  z_m → PCA 降维 → 条件去噪网络 → DDIM/Heun 采样 → PCA⁻¹ → Decoder → SDF_gen
```

**核心优势**：在 PCA 压缩后的 128~256 维空间上做扩散（而非直接在 256×256=65536 维像素空间），计算量降低约 128~256 倍。

### 1.3 五维条件向量

#### 子午面数据集 (single / F404)

| 字段 | 含义 | 单位 |
|------|------|------|
| `hub_r_end_mm` | 轮毂外径半径 | mm |
| `rim_r_start_mm` | 轮缘内径半径 | mm |
| `angle_web_deg` | 腹板倾角 | ° |
| `r_trans_bore_web_mm` | 孔-腹板过渡半径 | mm |
| `z_min` | 轴向下界 | mm |

#### 翼型数据集 (airfoil_uiuc_sdf)

| 字段 | 含义 | 单位 |
|------|------|------|
| `t_max` | 最大厚度 | 弦长比 |
| `x_tmax` | 最大厚度位置 | 弦长比 |
| `camber_max` | 最大弯度 | 弦长比 |
| `x_camber_max` | 最大弯度位置 | 弦长比 |
| `te_gap` | 尾缘间隙 | 弦长比 |

---

## 2. 核心架构：三阶段条件扩散管线

### 2.1 阶段一：自编码器 (Autoencoder)

```
输入: SDF(B,1,H,W) + Semantic Mask(B,1,H,W)                  [可选拼接]
╔══════════════════════════════════════╗
║ Encoder (4 层 ConvBlock + MaxPool2d) ║
║   256×256 → 128×128 → 64×64 → 32×32 → 16×16                ║
║   通道:    2→32→64→128→128                                   ║
║   潜向量: z_m ∈ ℝ^(B×64×16×16)                               ║
╚══════════════════════════════════════╝
╔══════════════════════════════════════╗
║ Decoder (4 层 ConvBlock + Upsample)  ║
║   16×16 → 32×32 → 64×64 → 128×128 → 256×256                ║
║   通道:   64→128→128→64→32→16→1                              ║
║   输出: SDF_recon ∈ ℝ^(B×1×256×256)                         ║
╚══════════════════════════════════════╝
```

- **ConvBlock**: `Conv2d→GroupNorm→SiLU→Conv2d→GroupNorm→SiLU`
- **潜维度**: 64×16×16 = 16,384 → 压缩比 ≈ 4× (256²/16384)
- **参数量**: ~4.5M

### 2.2 阶段二：质量门控 (LatentRepresentationGate)

在 AE 训练完成后自动运行，决定是否进入扩散阶段：

| 指标 | 默认阈值 | 含义 |
|------|---------|------|
| 重建 L1 | ≤ 0.08 | 每样本 AE 重建误差 |
| 潜空间 std | ≥ 0.05 | 全局潜变量标准差（防止坍缩） |
| 通过率 | ≥ 85% | 合格样本占总样本比例 |

门控报告写入 `latent_gate_report.json`，包含 `passed` (bool)、`pass_ratio`、逐样本明细。

### 2.3 阶段三：条件扩散 (Conditional Diffusion)

```
z_m(64×16×16) → flatten(16384) → PCA 投影 → w(128~256 维) → 归一化
                                      ↓
条件向量 c(5 维) ─────────────────→ 条件去噪网络 ←── 时间嵌入 t_emb
                                      ↓
                     预测噪声 ε_pred → DDIM/Heun 采样 → PCA⁻¹ → AE Decoder → SDF_gen
```

#### 关键技术

1. **PCA 降维**：SVD 分解训练集潜向量，取前 k 个主成分
   - 编码：`w = (x - mean)·V_k^T / std`
   - 解码：`x̂ = w·std·V_k + mean`

2. **Classifier-Free Guidance (CFG)**：
   - 训练：20% 概率随机丢弃条件向量
   - 推理：`ε = ε_uncond + cfg_scale·(ε_cond - ε_uncond)`

3. **DDIM 采样**：确定性跳步，200 步 → 50 步

4. **EDM 预条件化** (仅 edm_guided 方案)：Karras et al. 的 c_in/c_out/c_skip/c_noise 参数化 + Heun 二阶 ODE 采样

5. **PIDM 虚拟似然** (仅 pidm_guided 方案)：几何残差独立约束，目标 r→0

---

## 3. 六套方案详解

项目在 `schemes/` 下维护 **6 套完全独立的扩散方案**，每套方案代码不共享，可独立演进。

### 3.1 方案总表

| 方案 | `--scheme` | 去噪器 | PCA 维度 | 扩散框架 | 采样 | 条件校验 | 物理引导 | 独特输出 |
|------|-----------|--------|---------|----------|------|----------|----------|----------|
| **MLP 基线** | `mlp` | MLP (512×4层) | 128 | DDPM | DDIM 50步 | 无 | 可选 MSE对齐 | 基线指标 |
| **PCA-UNet** | `pca_unet` | 条件 UNet (2×8×8网格) | 128/256 | DDPM | DDIM 50步 | 无 | 无 | UNet 精度最高 |
| **尺寸引导** | `dim_guided` | MLP | 128 | DDPM | DDIM 50步 | 有 | 可选 | `condition_applicability.json` |
| **尺寸UNet** | `dim_unet` | 条件 UNet (4×6×8网格) | 192 | DDPM | DDIM 80步 | 有 | 可选 | 同上 + 192维UNet |
| **EDM引导** | `edm_guided` | EDM预条件化MLP | 128 | EDM | Heun 35步 | 有 | 可选 | Heun采样指标 |
| **PIDM物理** | `pidm_guided` | MLP | 128 | DDPM | DDIM 50步 | 有 | PIDM虚拟似然 | `pidm_config.json`、`mean_physics_residual` |

### 3.2 各方案详细说明

#### 3.2.1 `mlp` — MLP 基线扩散

- **去噪器**：`MLPDenoiser` — 4 层全连接（每层 512 维 + SiLU），输入拼接 `[w_t | c | t_emb]`
- **时间嵌入**：正弦位置编码 → Linear(128→128) → SiLU → Linear(128→128)
- **适用**：快速基线、方案比较基准
- **配置段名**：`diffusion`

#### 3.2.2 `pca_unet` — PCA-UNet 扩散

- **去噪器**：`ConditionalUNet` — 将 PCA 向量网格化后送入 3 层下采样 + 3 层上采样的 UNet
- **网格化**：`PcaGridSpec.from_dim(k)` 自动分解 k 维向量为 C×H×W（优先正方形，H/W 为 4 的倍数）
  - k=128 → 2×8×8；k=256 → 1×16×16
- **条件注入**：5 维条件向量广播到空间维度后与噪声 latent 拼接；FiLM 调制
- **适用**：高精度生成（UNet 捕捉空间结构）
- **配置段名**：`unet`
- **翼型实验**：本方案是 UIUC 翼型实验的唯一选型

#### 3.2.3 `dim_guided` — 尺寸引导 MLP 扩散

- **去噪器**：MLPDenoiser（同 `mlp`）
- **条件加载**：`ConditionVector` 支持 Excel (`.xlsx`)、NPZ、CSV 多级回退
- **适用性校验**：训练前评估各条件列的缺失率、方差、取值区间、物理约束违反数
- **适用**：真实工程数据（含条件校验）
- **配置段名**：`diffusion` + `condition_validation`

#### 3.2.4 `dim_unet` — 尺寸引导 UNet 扩散

- **去噪器**：ConditionalUNet（同 `pca_unet`），PCA 维度 192（4×6×8 网格）
- **条件加载**：ConditionVector（同 `dim_guided`）
- **AE 损失**：额外包含轮缘加权（rim）和梯度（gradient）损失
- **适用**：高精度 + 条件校验
- **配置段名**：`unet` + `condition_validation`

#### 3.2.5 `edm_guided` — EDM 预条件化扩散

- **去噪器**：`EDMMLPDenoiser` — EDM 预条件化包装的 MLP
  ```
  D_θ(x; σ) = c_skip(σ)·x + c_out(σ)·F_θ(c_in(σ)·x; c_noise(σ))
  ```
- **训练噪声**：`ln σ ~ N(P_mean=-1.2, P_std=1.2)`
- **采样**：Heun 二阶 ODE（梯形校正），35 步，ρ=7 幂律噪声序列
- **适用**：探索 EDM 框架效果
- **配置段名**：`diffusion` + `edm` + `condition_validation`

#### 3.2.6 `pidm_guided` — PIDM 物理信息扩散

- **去噪器**：MLPDenoiser（同 `mlp`）
- **训练损失**：`L = c_data·L_DDPM + c_residual·(-log p(r=0|x̂₀, var_t))`
- **几何残差分量**：
  - `r_thick = ReLU(min_thickness - estimate_thickness(SDF))` — 壁厚不足惩罚
  - `r_eikonal = w·|||∇SDF|| - 1|` — Eikonal 条件（SDF 梯度模长应为 1）
  - `r_area = |area_est - area_target|` — 可选面积约束
- **训练调度**：前 33% epoch 仅 DDPM，之后启用物理项
- **推理校正**（可选）：每步 x₀ 梯度校正
- **配置段名**：`diffusion` + `pidm` + `condition_validation`

### 3.3 方案选用决策树

```
需要条件校验？ ──Yes── 需要最高精度？ ──Yes── dim_unet
    │                        └─No─── 需要物理约束？ ──Yes── pidm_guided
    │                                        └─No──── dim_guided
    └─No── 需要高精度？ ──Yes── pca_unet
                └─No─── 需要 EDM？ ──Yes── edm_guided
                            └─No──── mlp (基线)
```

---

## 4. 统一入口 run.py

### 4.1 设计原则

所有方案通过 `run.py` 统一调度，`--scheme` 参数动态导入对应 `pipeline` 模块。每个方案的 `pipeline.py` 必须实现 `run_train()` 和 `run_test()` 两个函数。

### 4.2 子命令一览

| 子命令 | 用途 | 示例 |
|--------|------|------|
| `prepare-airfoil-uiuc` | UIUC 翼型数据预处理 | `python run.py prepare-airfoil-uiuc --force` |
| `prepare-splits` | 生成固定 train/test 划分 | `python run.py prepare-splits --dataset airfoil_uiuc_sdf --force` |
| `train` | 训练 AE + 扩散（可分阶段） | `python run.py train --scheme pca_unet --dataset airfoil_uiuc_sdf` |
| `test` | 评估 + 可视化 | `python run.py test --scheme pca_unet --run-dir outputs/.../<timestamp>` |
| `eval-airfoil-geometry` | 翼型几何校核与 .dat 导出 | `python run.py eval-airfoil-geometry --run-dir outputs/.../<timestamp> --candidates-per-condition 32` |
| `collect-timing` | 汇总所有 run 的耗时 | `python run.py collect-timing` |

### 4.3 train 子命令参数

```
python run.py train \
  --scheme {mlp|pca_unet|dim_guided|dim_unet|edm_guided|pidm_guided} \
  --dataset {single|F404|airfoil_uiuc_sdf} \
  [--stage {all|ae|diff|unet}] \     # 分阶段训练
  [--fast] \                          # 冒烟测试（减少 epoch）
  [--ae-run-dir <path>]               # 扩散阶段指定已有 AE 目录
```

阶段别名规则：
- MLP 类方案（`mlp`/`dim_guided`/`edm_guided`/`pidm_guided`）：`--stage unet` 自动归一化为 `--stage diff`
- UNet 类方案（`pca_unet`/`dim_unet`）：`diff` 与 `unet` 等效，均为 UNet 扩散阶段

---

## 5. 数据管线

### 5.1 输入：NPZ 样本格式

每个样本为单个 `.npz` 文件：

```
sample_id.npz
├── sdf2d_norm           float32[H, W]    归一化 SDF（正值=外部，负值=实体）
├── semantic_mask        float32[H, W]    语义分区（子午面: hub≈0.2, web≈0.6, rim≈0.9）
├── coords_raw           float32[N, 2]    原始翼型坐标（翼型数据集）
├── coords_resampled     float32[2, N]    重采样后坐标（翼型数据集）
├── condition_raw        float32[K]       几何条件数组（翼型数据集）
├── condition_json       str (JSON)       五维设计参数（子午面数据集，可选）
├── engineering_curves_json  str (JSON)   工程曲线（可推导条件，可选）
└── physics              dict            物理属性（min_thickness_mm, area_mm2）
```

### 5.2 数据集组织

```
data/
├── single/                              # 单样本子午面（五维条件方差充足）
│   ├── dataset.json                     # 数据集元信息
│   └── processed/
│       ├── meridian_index.csv
│       ├── train_split.csv / test_split.csv
│       ├── condition_stats.json
│       └── meridian_samples/*.npz
│
├── F404/                                # F404 子午面（部分条件列为常量）
│   ├── dataset.json
│   └── processed/
│       ├── train_split.csv / test_split.csv
│       ├── condition_specs.jsonl        # 补全 NPZ 中缺失的条件列
│       └── meridian_samples/*.npz
│
├── airfoil_uiuc_sdf/                    # UIUC 翼型 SDF 数据集
│   ├── dataset.json
│   └── processed/
│       ├── airfoil_index.csv            # 1646 条全量索引
│       ├── train_split.csv              # 1399 条训练集
│       ├── test_split.csv               # 247 条测试集 (15%)
│       ├── condition_stats.json
│       ├── filter_report.json           # 19 个过滤样本的原因
│       ├── split_meta.json
│       └── airfoil_samples/*.npz        # 1646 个 NPZ
│
└── airfoil/raw/uiuc/
    └── coord_seligFmt/                  # UIUC 原始 .dat 翼型库 (~1665 个)
```

### 5.3 dataset.json 结构

```json
{
  "processed_dir": "processed",
  "index_file": "airfoil_index.csv",
  "npz_dir": "airfoil_samples",
  "condition_columns": ["t_max", "x_tmax", "camber_max", "x_camber_max", "te_gap"],
  "split": {
    "test_size": 0.15,
    "seed": 42,
    "filter_passed_quality": true
  },
  "param_table_path": "data/.../param_table.csv",          // 可选
  "condition_excel_path": "data/.../params.xlsx"           // 可选 (dim_* 方案)
}
```

### 5.4 条件加载优先级（dim_guided / dim_unet / edm_guided / pidm_guided）

```
1. train_split.csv / test_split.csv 索引行中的条件列
2. 内存参数表 (CSV 或 Excel)
3. dataset.json 中配置的 Excel/CSV 参数表路径
4. NPZ 内 condition_json 或 engineering_curves_json 推导
```

### 5.5 UIUC 翼型预处理流程

```
UIUC .dat 文件
    │
    ├─ parse_selig_dat()         → 解析坐标，跳过标题行/空行
    ├─ normalize_airfoil()       → 平移+缩放至弦长=1，前缘 x=0
    ├─ split_surfaces()          → 拆分为 upper/lower 表面
    ├─ resample_surfaces()       → cosine x_grid (257点) 重采样
    ├─ compute_airfoil_conditions() → 计算 5 维几何条件
    ├─ validate_airfoil()        → 合法性检查（厚度/弯度/尾缘范围）
    │
    ├─ 过滤: t_max∉[0.02,0.35], 插值失败, 坐标异常等 → 19 个被过滤
    │
    ├─ build_airfoil_polygon()   → 构建闭合多边形
    ├─ rasterize_airfoil_sdf()   → 栅格化 SDF (128×256)
    │
    └─ 输出: sample_id.npz
        ├── sdf2d_norm (128×256)
        ├── semantic_mask (128×256)
        ├── coords_raw
        ├── coords_resampled
        ├── condition_raw (5维)
        └── condition_json
```

---

## 6. 网络架构详解

### 6.1 自编码器 (MeridianAutoEncoder)

#### Encoder 结构

```
输入: [B, C_in, H, W]     (C_in=1 或 2，取决于 use_semantic)
    ↓
ConvBlock(C_in → 32)       # Conv2d(3×3)→GN→SiLU→Conv2d(3×3)→GN→SiLU + MaxPool2d(2)
    ↓  [B, 32, H/2, W/2]
ConvBlock(32 → 64) + MaxPool2d(2)
    ↓  [B, 64, H/4, W/4]
ConvBlock(64 → 128) + MaxPool2d(2)
    ↓  [B, 128, H/8, W/8]
ConvBlock(128 → 128) + MaxPool2d(2)
    ↓  [B, 128, H/16, W/16]
Conv2d(128 → latent_channels=64, kernel=1×1)
    ↓
z_m: [B, 64, H/16, W/16]    潜向量
```

#### Decoder 结构

```
z_m: [B, 64, H/16, W/16]
    ↓
Conv2d(64 → 128, 1×1) + Upsample(×2)
    ↓  [B, 128, H/8, W/8]
ConvBlock(128 → 128) + Upsample(×2)
    ↓  [B, 128, H/4, W/4]
ConvBlock(128 → 64) + Upsample(×2)
    ↓  [B, 64, H/2, W/2]
ConvBlock(64 → 32) + Upsample(×2)
    ↓  [B, 32, H, W]
ConvBlock(32 → 16)
    ↓
Conv2d(16 → 1, 3×3, padding=1)  → SDF_recon [B, 1, H, W]
```

#### 关键参数

| 参数 | 子午面默认 | 翼型默认 | 说明 |
|------|----------|---------|------|
| `in_channels` | 2 (SDF + mask) | 1 (仅 SDF) | 输入通道数 |
| `latent_channels` | 64 | 64 | 潜向量通道数 |
| `input_size` | [256, 256] | [128, 256] | 输入尺寸（必须是 latent_spatial 的 16 倍） |
| `latent_spatial` | [16, 16] | [8, 16] | 潜空间尺寸 |
| 潜维度 | 16,384 | 8,192 | latent_channels × H_lat × W_lat |

### 6.2 MLP 去噪器 (MLPDenoiser)

用于 `mlp` / `dim_guided` / `edm_guided` / `pidm_guided` 方案。

```
输入:
  x_t  ∈ ℝ^(B × pca_dim)                  带噪潜向量
  t    ∈ ℤ^B                               离散时间步
  c    ∈ ℝ^(B × 5)                         条件向量

时间嵌入:  SinusoidalPosEmb(t, dim=128) → Linear(128→128) → SiLU → Linear(128→128)
拼接:      h = [x_t | c | t_emb] ∈ ℝ^(B × (pca_dim + 5 + 128))

MLP 主体:
  Linear(pca_dim+5+128 → 512) → SiLU
  Linear(512 → 512) → SiLU
  Linear(512 → 512) → SiLU
  Linear(512 → pca_dim)

输出: ε_pred ∈ ℝ^(B × pca_dim)            预测噪声
```

### 6.3 PCA-UNet 去噪器 (ConditionalUNet)

用于 `pca_unet` / `dim_unet` 方案。

#### 网格化 (PcaGridSpec)

| pca_dim | 网格分解 | 使用方案 | 说明 |
|---------|---------|---------|------|
| 128 | C=2, H=8, W=8 | pca_unet (子午面) | |
| 192 | C=4, H=6, W=8 | dim_unet | |
| 256 | C=1, H=16, W=16 | pca_unet (翼型) | |

#### UNet 架构

```
输入: x_t ∈ ℝ^(B×C×H×W), c ∈ ℝ^(B×5), t ∈ ℤ^B

条件嵌入:
  t_emb = SinusoidalPosEmb(t) → Linear → SiLU → Linear → 128
  c_emb = Linear(5→128) → SiLU → Linear(128→128)
  emb = [t_emb | c_emb] ∈ ℝ^(B × 256)

空间条件广播:
  c_spatial = c.unsqueeze(-1,-1).expand(B, 5, H, W)
  x_in = [x_t | c_spatial] ∈ ℝ^(B × (C+5) × H × W)

Encoder:
  in_conv: Conv2d(C+5 → base_ch, 3×3) → ResBlock → skip_1   [B, base_ch, H, W]
  down1:   Conv2d(base_ch → 2·base_ch, stride=2) → ResBlock → skip_2  [B, 2·base_ch, H/2, W/2]
  down2:   Conv2d(2·base_ch → 4·base_ch, stride=2) → ResBlock_mid1    [B, 4·base_ch, H/4, W/4]
  mid:     ResBlock_mid2

Decoder:
  up2:     ConvTranspose2d(4·base_ch → 2·base_ch) → concat(skip_2) → ResBlock
  up1:     ConvTranspose2d(4·base_ch → base_ch) → concat(skip_1) → ResBlock
  out_conv: Conv2d(2·base_ch → C, 3×3)

输出: ε_pred ∈ ℝ^(B × C × H × W)
```

#### ResBlock（含 FiLM 调制）

```
  h = GroupNorm → SiLU → Conv2d
  h = GroupNorm(h)
  scale, shift = Linear(emb)           # FiLM 调制
  h = h * (1 + scale_spatial) + shift_spatial
  h = SiLU(h) → Conv2d
  output = x + h                       # 残差连接
```

### 6.4 EDM 预条件化去噪器 (EDMMLPDenoiser)

用于 `edm_guided` 方案，是对 `MLPDenoiser` 的预条件化包装：

```
c_skip(σ) = σ_data² / (σ² + σ_data²)
c_out(σ)  = σ · σ_data / √(σ² + σ_data²)
c_in(σ)   = 1 / √(σ² + σ_data²)
c_noise(σ) = ¼ · ln(σ)

D_θ(x; σ) = c_skip(σ)·x + c_out(σ)·F_θ(c_in(σ)·x; c_noise(σ))
```

其中 `F_θ` 是原始 MLPDenoiser，`σ_data` 从训练集 PCA 编码自动估计或手动指定。

### 6.5 PCA 降维 (LatentPCA)

```
拟合:
  Z_train ∈ ℝ^(N × D)                      D = latent_channels × H_lat × W_lat
  mean = Z_train.mean(0)                   ∈ ℝ^D
  Z_centered = Z_train - mean
  U, S, Vt = SVD(Z_centered)
  components = Vt[:k, :]                   ∈ ℝ^(k × D)
  proj = Z_centered @ components.T         ∈ ℝ^(N × k)
  std = proj.std(0)                        ∈ ℝ^k   (全局归一化因子)

编码 (encode):
  w = ((z - mean) @ components.T) / std    ∈ ℝ^k

解码 (decode):
  z_hat = (w * std) @ components + mean    ∈ ℝ^D
  → reshape  → (latent_channels, H_lat, W_lat)
```

---

## 7. 损失函数体系

### 7.1 自编码器损失

```
L_AE = w_l1·L1 + w_l2·L2 + w_zero·L_zero + w_dim·L_dim + w_sem·L_sem
```

| 损失项 | 默认权重 | 公式 | 用途 |
|--------|---------|------|------|
| L1 | 1.0 | `｜recon - target｜₁` | 全局 SDF 重建 |
| L2 | 0.5 | `(recon - target)²` | 像素级精度 |
| zero_level | 2.0→3.0 (翼型) | `Σ w(x,y)·｜recon-target｜ / Σ w(x,y)` | 边界加权（高斯衰减 σ=1.5mm） |
| dimension | 0.3→0.0 (翼型) | `｜thickness_est - gt｜ + ｜area_est - gt｜/10000` | 物理尺寸约束 |
| semantic | 0.2→0.0 (翼型) | `(1 - coverage)²` | hub/web/rim 语义覆盖率 |

**零等值面加权**：
```
w(x,y) = exp(-|SDF(x,y) · sdf_scale| / sigma_mm)
L_zero = Σ w(x,y) · |recon(x,y) - target(x,y)| / Σ w(x,y)
```

### 7.2 扩散损失

#### DDPM 噪声预测 (mlp / pca_unet / dim_* / pidm_guided)

```
L_DDPM = MSE(ε_pred, ε)        ε ~ N(0, I)
```

#### EDM 预条件化 (edm_guided)

```
L_EDM = MSE(F_θ(c_in·x_t; c_noise), (x₀ - c_skip·x_t) / c_out)    等效于加权的去噪 MSE，λ(σ) ∝ 1/c_out²
```

#### PIDM 虚拟似然 (pidm_guided)

```
L = c_data · L_DDPM + c_residual · (-log p(r=0 | x̂₀, var_t))

其中:
  r = geometry_residual(sdf_hat)          # 壁厚违反 + Eikonal + (可选) 面积
  var_t = posterior_variance[t]            # 与扩散步绑定的后验方差

训练调度: 前 warmup_frac (默认33%) epoch 仅 L_DDPM，之后启用物理项
```

#### Physics Guidance (mlp / dim_* / edm_guided，默认关闭)

```
L_physics = w_pg · MSE(risk_pred, risk_gt)
           = w_pg · MSE(1/(thickness_est+ε), 1/(gt_thickness+ε))
```

注意：这是"对齐 GT 风险代理"而非独立物理约束，与 PIDM 的 r→0 方式不同。

---

## 8. 扩散模型训练与采样

### 8.1 训练循环

```python
for epoch in range(epochs):
    for batch in train_loader:
        # 1. 前向扩散
        t ~ Uniform(1, T)
        ε ~ N(0, I)
        x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε

        # 2. CFG 条件丢弃
        cond_mask = (rand(B) < cfg_dropout)      # 20% 概率丢弃条件
        c_masked = c * ~cond_mask

        # 3. 去噪
        ε_pred = denoiser(x_t, t, c_masked)

        # 4. 噪声预测损失
        loss = MSE(ε_pred, ε)

        # 5. (可选) 物理引导
        if physics_enabled and epoch > warmup_epochs:
            x̂₀ = predict_x0(x_t, ε_pred)
            sdf_hat = AE.decode(PCA⁻¹(x̂₀))
            loss += w_physics · physics_loss(sdf_hat)

        # 6. 优化
        loss.backward()
        clip_grad_norm(max_norm=1.0)
        optimizer.step()

    # 验证: 以 gen_l1 为最优模型保存依据
    val_gen_l1 = evaluate(val_conditions)
    if val_gen_l1 < best_gen_l1:
        save(best_diffusion.pt)
```

### 8.2 DDIM 采样 (mlp / pca_unet / dim_* / pidm_guided)

```
x_T ~ N(0, I)
timestep_indices = linspace(T-1, 0, sample_steps)          # e.g. 50 steps

for t, t_prev in zip(timestep_indices[:-1], timestep_indices[1:]):
    ε_c = denoiser(x_t, t, c)                               # 条件预测
    ε_u = denoiser(x_t, t, 0)                               # 无条件预测
    ε   = ε_u + cfg_scale · (ε_c - ε_u)                     # CFG 混合

    x̂₀     = (x_t - √(1-ᾱ_t)·ε) / √ᾱ_t                     # 预测 x₀
    x_{t-1} = √ᾱ_{t-1}·x̂₀ + √(1-ᾱ_{t-1} - σ²_t)·ε          # DDIM 确定性步进 (η=0)

sdf_gen = AE.decode(PCA⁻¹(x₀))
```

### 8.3 Heun 二阶采样 (edm_guided)

```
σ_seq: σ_max > σ_1 > ... > σ_min > 0                      # ρ=7 幂律分布

x_t ~ N(0, σ_max²·I)

for i in range(N):
    d_cur   = (x_t - D_θ_cfg(x_t; σ_i)) / σ_i              # score 估计
    x_euler = x_t + (σ_{i+1} - σ_i) · d_cur                 # Euler 步

    if σ_{i+1} == 0:
        x_t = D_θ_cfg(x_euler; 0)                           # 最后一步直接去噪
    else:
        d_next = (x_euler - D_θ_cfg(x_euler; σ_{i+1})) / σ_{i+1}
        x_t = x_t + (σ_{i+1} - σ_i) · ½(d_cur + d_next)    # 梯形校正
```

### 8.4 x₀ 梯度校正 (pidm_guided，可选)

```python
for each DDIM step:
    x̂₀ = predict_x0(x_t, ε_pred)
    sdf_hat = AE.decode(PCA⁻¹(x̂₀))
    r = geometry_residual(sdf_hat)

    for _ in range(n_correction):              # 每步 n 次梯度校正
        grad = ∂r/∂x̂₀
        x̂₀ = x̂₀ - correction_step_size · grad

    # 继续 DDIM 步进
```

---

## 9. 输出产物与目录规范

### 9.1 运行目录结构

```
outputs/{scheme}/{dataset}/{timestamp}/
├── logs/
│   └── train.log                              # 完整训练日志（含 stdout）
├── timing.json                                 # 训练/测试各阶段耗时
├── autoencoder/                               # AE 阶段产物
│   ├── best_autoencoder.pt                    # 最优 AE 权重（含 in_channels, epoch）
│   ├── condition_stats.json                   # 条件归一化统计（mean, std, columns）
│   ├── summary.json                           # AE 训练汇总（gate_passed, best_val_l1）
│   ├── training_history.json                 # 逐 epoch 损失数组
│   ├── training_dashboard.png                # 训练面板（损失曲线 + 重建对比）
│   ├── training_curves.png                   # 损失曲线
│   ├── body_latent_train.json                # 训练集体潜码记录 (z_m, condition, gate_passed)
│   ├── body_latent_test.json                 # 测试集体潜码记录
│   ├── latent_gate_report.json               # 质量门控报告 (passed, pass_ratio, 逐样本)
│   ├── latent_gate_report.png                # 门控可视化
│   ├── latent_pca_test.png                   # 潜空间 PCA 可视化
│   └── visualizations/                       # 重建对比图
│       ├── reconstructions_test/             # 逐样本三联图 (GT/SDF/SDF_recon)
│       ├── grid_test.png                     # 样本网格对比
│       └── metrics_test.json                 # 逐样本重建指标
│
├── diffusion/                                 # 扩散阶段产物
│   ├── best_{diffusion|unet_diffusion}.pt     # 最优去噪器权重
│   ├── latent_pca.json                        # PCA 参数 (mean, components, std, dim)
│   ├── summary.json                           # 扩散汇总 (mean_gen_l1, pca_dim, denoiser)
│   ├── training_history.json                 # 逐 epoch 损失
│   ├── training_dashboard.png                # 训练面板
│   ├── training_curves.png                   # 损失曲线
│   ├── generation_metrics.json               # 逐样本生成 L1 指标
│   ├── grid_test.png                         # 生成样本网格对比
│   ├── metrics_test.png                      # 逐样本 L1 散点图
│   └── generations/                          # 逐样本 SDF 生成图
│       └── <sample_id>.png
│
├── visualizations/                            # 汇总报告 (run.py test 生成)
│   ├── summary_report.json                    # 综合报告（AE 门控 + 扩散 L1 + 计时）
│   ├── timing.json                            # 计时报告副本
│   ├── test_generation_l1.png                 # 逐样本生成 L1 分布
│   ├── test_ae_recon_l1.png                   # 逐样本 AE 重建 L1 分布
│   ├── test_gen_vs_ae_recon.png               # 生成 vs AE 重建并排对比
│   └── generations/                           # 逐样本 SDF 对比图（GT vs Gen）
│
├── geometry_eval_rerank_c{1|32}/              # 翼型几何校核与重排 (eval-airfoil-geometry)
│   ├── rerank_summary.csv                     # 重排汇总（条件误差排序）
│   ├── per_sample_geometry.csv               # 逐样本完整几何指标
│   ├── geometry_eval_report.json              # 评估报告
│   ├── best_airfoils_for_aero.md              # 气动校核推荐列表
│   ├── candidate_dat/{sample_id}/             # 全部候选 .dat 翼型文件
│   └── top_dat/{sample_id}.dat                # 最优候选 .dat 翼型文件
│
└── xfoil_eval_top{N}_re{Re}_m{Mach}_a{A1}_{A2}/  # XFOIL 气动校核结果
    ├── xfoil_report.md                        # Markdown 详细报告
    ├── xfoil_report.json                      # JSON 结构化报告
    ├── xfoil_conclusion_cn.md                 # 中文结论报告
    ├── xfoil_summary.csv                      # 全部 case 汇表
    ├── xfoil_paired_comparison.csv            # 生成 vs 原始 配对对比
    └── visualizations/                        # 校核可视化
        ├── xfoil_convergence_summary.png
        ├── xfoil_best_ld_distribution.png
        ├── xfoil_paired_metric_scatter.png
        ├── xfoil_top_generated_best_ld.png
        ├── xfoil_top_generated_polar_curves.png
        └── xfoil_top_generated_airfoil_shapes.png
```

### 9.2 关键 JSON 结构

#### summary_report.json

```json
{
  "scheme": "pca_unet",
  "dataset": "airfoil_uiuc_sdf",
  "run_dir": "outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115",
  "gate": {
    "passed": true,
    "pass_ratio": 0.92,
    "mean_recon_l1_train": 0.006,
    "mean_recon_l1_test": 0.008
  },
  "diffusion": {
    "mean_gen_l1": 0.0374,
    "best_val_gen_l1": 0.027,
    "mean_ae_recon_l1": 0.00355,
    "pca_dim": 256,
    "denoiser": "pca_unet"
  },
  "timing": {
    "train": { "total_sec": 1734, "total_human": "28m 54s", "stages": {...} },
    "test": { "total_sec": 5.4, "total_human": "5.4s", "stages": {...} }
  },
  "artifacts": [
    "diff_training_dashboard.png", "diff_grid_test.png", ...
  ]
}
```

#### timing.json

```json
{
  "scheme": "pca_unet",
  "dataset": "airfoil_uiuc_sdf",
  "run_dir": "...",
  "train": {
    "stage_requested": "unet",
    "fast": false,
    "started_at": "2026-06-15T11:41:15+08:00",
    "finished_at": "2026-06-15T12:10:09+08:00",
    "total_sec": 1733.971,
    "total_human": "28m 54.0s",
    "stages": { "diffusion": { "sec": 1733.971, "human": "28m 54.0s" } }
  },
  "test": {
    "started_at": "2026-06-15T13:01:43+08:00",
    "finished_at": "2026-06-15T13:01:48+08:00",
    "total_sec": 5.375,
    "total_human": "5.4s"
  }
}
```

---

## 10. UIUC 翼型条件扩散实验

### 10.1 实验概览

本次实验将子午面条件扩散管线改造为翼型条件扩散生成，使用 UIUC Airfoil Coordinates Database (~1665 个翼型) 作为训练数据。

### 10.2 实施时序

```
2026-06-15 09:xx  ── commit 8c57981 翼型几何核心 (解析/归一化/条件计算)
2026-06-15 10:xx  ── commit 981eb13 翼型 SDF 栅格化
2026-06-15 11:0x  ── commit a2f530a UIUC 预处理流水线 + 数据集配置
2026-06-15 11:06  ── run.py prepare-airfoil-uiuc --force  → 1646/1665 通过
2026-06-15 11:xx  ── commit 262676f pca_unet 数据集 smoke test
2026-06-15 11:0x  ── run.py train --scheme pca_unet --stage ae (GPU)  → AE 训练
2026-06-15 11:41  ── run.py train --scheme pca_unet --stage unet (GPU)  → 扩散训练
2026-06-15 12:10  ── 扩散训练完成 (28分54秒)
2026-06-15 13:01  ── run.py test → 评估/可视化
2026-06-15 13:xx  ── commit 94f379b 几何校核重排
2026-06-15 13:xx  ── run.py eval-airfoil-geometry --candidates-per-condition 32
2026-06-15 14:xx  ── 几何重排完成
2026-06-15 15:xx  ── commit 9a6f5f8 XFOIL 批量校核
2026-06-15 15:xx  ── XFOIL top50 校核 + 可视化
2026-06-15 16:xx  ── commit 4155620/01121ba 文档 + 可视化
```

### 10.3 各阶段关键指标

#### 数据阶段

| 指标 | 值 |
|------|-----|
| 原始 UIUC 文件 | ~1665 个 .dat |
| 通过过滤 | 1646 (98.9%) |
| 失败/跳过 | 19 (t_max∉[0.02,0.35], 插值失败, 坐标异常等) |
| 训练集 | 1399 (85%) |
| 测试集 | 247 (15%) |
| SDF 分辨率 | 128 × 256 |
| 配置 AE input_size | [128, 256], latent_spatial=[8, 16] |

#### AE 训练 (`20260615_110639`)

| 指标 | 值 |
|------|-----|
| 方案 | pca_unet |
| 架构 | Conv encoder/decoder, 4 层下采样 |
| 输入 | 单通道 SDF (use_semantic=false) |
| Epochs | 80 (full) |
| 最终 val L1 | **0.00354** |
| 几何 pass ratio | **100%** |
| 模型权重 | `best_autoencoder.pt` (4.0 MB) |

#### 扩散训练 (`20260615_114115`)

| 指标 | 值 |
|------|-----|
| 去噪器 | ConditionalUNet |
| PCA 维度 | 256 (1×16×16 网格) |
| UNet base_ch | 384 |
| Batch size | 1024 |
| Epochs | 200 (full) |
| CFG | dropout=0.2, scale=1.5 |
| DDIM 采样步数 | 50 |
| 训练时间 | **28分54秒** |
| 测试集 gen L1 | **0.0374** |
| best gen L1 | **0.0246** |

#### 几何重排 (`geometry_eval_rerank_c32`)

| 指标 | 值 |
|------|-----|
| 每条件候选数 | 32 |
| 总候选翼型 | 247 × 32 = 7904 |
| Top 数据 | 每个条件选出 1 个最优 (条件误差最小) |
| 导出格式 | Selig .dat (321 点, cosine 257 + 闭合) |

**Top 5 几何最优**：

| rank | sample_id | score | Δt_max | Δcamber | Δte_gap |
|---:|---:|---:|---:|---:|---:|
| 1 | dga1138 | 0.702 | 0.0003 | 0.0030 | 0.0059 |
| 2 | goe188 | 0.851 | 0.0006 | 0.0047 | 0.0045 |
| 3 | fg2 | 0.936 | 0.0038 | 0.0029 | 0.0121 |
| 4 | m14 | 1.032 | 0.0001 | 0.0051 | 0.0090 |
| 5 | n6h10 | 1.206 | 0.0002 | 0.0047 | 0.0088 |

#### XFOIL 气动校核 (`xfoil_eval_top50_re1e6_m010_a-2_6`)

| 指标 | 生成翼型 | UIUC 原始同名翼型 |
|------|---------:|-----------------:|
| 有效 polar | 40/50 | 41/50 |
| 完整 alpha sweep (-2°~6°, 9点) | 9/50 | 23/50 |
| 平均 best L/D | 30.89 | **102.71** |
| 中位 best L/D | 23.54 | 96.34 |
| 平均 max CL | 0.553 | 0.921 |
| 有效配对 (32对) 平均 ΔL/D | -73.42 | — |

**XFOIL Top 8 生成翼型**：

| sample_id | best L/D | max CL | 收敛 |
|---|---:|---:|---:|
| goe188 | 103.41 | 0.930 | 9/9 |
| m14 | 94.14 | 0.915 | 9/9 |
| fx74cl6140 | 88.47 | 1.082 | 9/9 |
| goe346 | 76.16 | 0.224 | 4/9 |
| goe322 | 74.69 | 1.065 | 8/9 |
| ua2-180 | 57.16 | 0.879 | 9/9 |
| pmc19sm | 55.77 | 0.863 | 8/9 |
| rc08n1 | 49.66 | 0.781 | 9/9 |

---

## 11. XFOIL 气动校核子系统

### 11.1 构建与部署

| 项目 | 值 |
|------|-----|
| 源码 | `https://github.com/RobotLocomotion/xfoil` (commit `d11a1544`) |
| 可执行 | `/home/vipuser/tools/xfoil/build/src/xfoil-6.97` |
| Conda 环境 | `/home/vipuser/conda_envs/xfoil-build` |
| 编译器 | gfortran (conda-forge) + CMake |
| 依赖 | xorg-libx11, xorg-xorgproto (X11 头文件) |

### 11.2 批量校核脚本能力

`scripts/airfoil/run_xfoil_batch.py`：

| 能力 | 说明 |
|------|------|
| 自动配对 | 生成翼型与 UIUC 原始同名翼型配对 |
| 多进程 | CPU 并行 (--workers N)，默认 8 |
| 无头运行 | 自动 `PLOP\nG\n\n` 关闭 X11 图形 |
| 几何清洗 | cosine 重采样 / camber-thickness 平滑 / 尾缘渐变 / 最小厚度夹紧 |
| 输出 | polar.dat / polar.csv / xfoil_summary.csv / xfoil_report.md / xfoil_report.json |
| 超时保护 | 每个 case 独立超时 (--timeout 35~45s) |

### 11.3 XFOIL 校核工况

```
Re     = 1,000,000
Mach   = 0.10
Ncrit  = 9
Alpha  = -2° to 6°, step 1° (共 9 个攻角)
Max iter = 120
几何清洗: cosine 161 点/面, 平滑 8 次, 尾缘 x/c=0.90→1.00 渐变到 0.001c
```

### 11.4 XFOIL 实验中的工程问题与解决方案

| 问题 | 现象 | 解决方案 |
|------|------|----------|
| 长路径读取失败 | XFOIL `LOAD <long/path/airfoil.dat>` 解析成单字符 | 每个 case 独立工作目录 + 短文件名 `airfoil.dat` |
| 无 DISPLAY 崩溃 | `Cannot open display...aborting` | 批处理开头插入 `PLOP\nG\n\n` 关闭图形模式 |
| 翼型点数超限 | 513 点超 XFOIL 节点上限，弦长/厚度识别异常 | cosine 重采样至 161 点/面 (共 321 点) |
| Viscous BL 不收敛 | SDF 轮廓局部锯齿/尖角导致 NaN | camber/thickness 平滑 + 尾缘渐变 + 最小厚度夹紧 |
| 部分原始翼型也不收敛 | UIUC .dat 带粗糙尾缘或异常点列 | 同样接受清洗，41/50 原始翼型收敛 |

---

## 12. 代码组织全景图

```
Meri-FID-Gate-Airfoil/                                ← 项目根
│
├── run.py                                            ← ★ 统一 CLI 入口
│
├── scripts/                                          ← 方案无关的工具脚本
│   ├── airfoil_geometry.py                           ← 翼型几何核心 (解析/归一化/条件)
│   ├── airfoil_sdf.py                                ← SDF 栅格化 (多边形→距离场)
│   ├── prepare_airfoil_uiuc.py                       ← UIUC 批量预处理编排
│   ├── prepare_splits.py                             ← train/test 固定划分
│   ├── split_utils.py                                ← 划分工具
│   ├── timing_utils.py                               ← 训练/测试计时 + collect-timing
│   ├── airfoil/                                      ← 翼型实验专属脚本
│   │   ├── run_xfoil_batch.py                        ← XFOIL 批量校核
│   │   ├── evaluate_generated_airfoils.py            ← 几何校核 + rerank
│   │   └── plot_xfoil_results.py                     ← XFOIL 结果可视化
│   └── __pycache__/
│
├── data/                                             ← 数据集
│   ├── single/                                       ← 子午面 single
│   ├── F404/                                         ← 子午面 F404
│   ├── airfoil_uiuc_sdf/                             ← 翼型 SDF
│   └── airfoil/raw/uiuc/coord_seligFmt/              ← UIUC 源码 .dat
│
├── schemes/                                          ← 6 套独立扩散方案
│   ├── mlp/           {pipeline,models,train,losses,gate,records,utils,data,config}
│   ├── pca_unet/      {pipeline,models,train,losses,gate,records,utils,data,config}
│   ├── dim_guided/    {pipeline,models,train,losses,gate,records,utils,data,config}
│   ├── dim_unet/      {pipeline,models,train,losses,gate,records,utils,data,config}
│   ├── edm_guided/    {pipeline,models,train,losses,gate,records,utils,data,config}
│   └── pidm_guided/   {pipeline,models,train,losses,gate,records,utils,data,physics,config}
│
├── tests/                                            ← 测试 (19 airfoil + 子午面)
│   ├── test_airfoil_geometry.py                      ← 翼型几何核心测试
│   ├── test_airfoil_sdf.py                           ← SDF 栅格化测试
│   ├── test_prepare_airfoil_uiuc.py                  ← 预处理流水线测试
│   ├── test_pca_unet_airfoil_dataset.py              ← 数据集读取测试
│   ├── test_pca_unet_airfoil_smoke.py                ← AE CPU smoke test
│   ├── test_pca_unet_airfoil_diffusion_config.py     ← 扩散配置校验
│   ├── test_airfoil_cleanup_rerank.py                ← 几何重排测试
│   └── test_field_pipeline.py / test_scalar_pipeline.py ... (子午面)
│
├── outputs/                                          ← 所有训练产物
│   ├── mlp/{single,F404}/
│   ├── pca_unet/{F404,airfoil_uiuc_sdf}/
│   ├── dim_guided/{single,F404}/
│   ├── dim_unet/{single,F404}/
│   ├── edm_guided/{single,F404}/
│   ├── pidm_guided/{single,F404}/
│   ├── timing_summary.json / timing_summary.csv / timing_comparison.png
│   └── _timing_smoke/
│
├── docs/                                             ← 项目文档
│   ├── airfoil_experiment_steps.md                   ← 翼型实验步骤指南
│   ├── airfoil_xfoil_experiment_record.md            ← XFOIL 实验完整记录
│   └── airfoil_gpu_runbook.md                        ← GPU 机器操作手册
│
├── wenxian/                                          ← 参考文献
│   ├── Karras et al. - Elucidating EDM (NeurIPS 2022).pdf
│   └── Karras et al. - Analyzing Training Dynamics of Diffusion (2024).pdf
│
├── 实验方案.md                                        ← UIUC 翼型实验总体设计
├── 编码计划.md                                        ← 五阶段编码实施计划
├── PROJECT_DESCRIPTION.md                            ← ★ 本文件
├── README.md                                         ← 项目 README
└── requirements.txt
```

### 每个方案的内部结构（以 pca_unet 为例）

```
schemes/pca_unet/
├── __init__.py               ← scheme_name() 标识
├── pipeline.py               ← run_train() / run_test() 入口
├── test.py                   ← run_eval() 评估逻辑
│
├── config/                   ← 各数据集超参 JSON
│   ├── single.json
│   ├── F404.json
│   └── airfoil_uiuc_sdf.json
│
├── models/                   ← 网络定义
│   ├── autoencoder.py        ← MeridianAutoEncoder
│   ├── latent_pca.py         ← LatentPCA (SVD + 编解码)
│   ├── pca_unet.py           ← ConditionalUNet (PCA网格 + 条件UNet)
│   └── diffusion.py          ← DiffusionSchedule (DDPM β schedule)
│
├── train/                    ← 训练循环
│   ├── train_ae.py           ← AE 训练 + 潜码编码 + Gate
│   ├── train_diffusion.py    ← UNet 扩散训练
│   └── diffusion_codec.py    ← DiffusionLatentCodec (PCA+AE 编解码)
│
├── losses/
│   └── ae_losses.py          ← L1/L2/零等值面/尺寸/语义 损失
│
├── gate/
│   └── latent_gate.py        ← 潜空间质量门控
│
├── records/
│   └── body_latent.py        ← BodyLatentRecord 数据结构
│
├── utils/
│   ├── config.py             ← load_config()
│   ├── paths.py              ← project_root / make_run_dir
│   └── visualization.py      ← 训练曲线/对比图/save_json
│
└── data/
    ├── dataset.py            ← MeridianSDFDataset (NPZ → Tensor)
    └── splits.py             ← 划分加载
```

---

## 13. 配置文件参考

### 13.1 方案配置完整示例 (pca_unet / airfoil_uiuc_sdf)

```json
{
  "dataset": "airfoil_uiuc_sdf",
  "autoencoder": {
    "in_channels": 1,
    "latent_channels": 64,
    "latent_spatial": [8, 16],
    "input_size": [128, 256],
    "epochs": 100,
    "batch_size": 16,
    "lr": 0.0003,
    "use_semantic": false,
    "loss_weights": {
      "l1": 1.0,
      "l2": 0.5,
      "zero_level": 3.0,
      "zero_sigma_mm": 1.0,
      "dimension": 0.0,
      "semantic": 0.0
    }
  },
  "latent_gate": {
    "max_recon_l1": 0.08,
    "min_latent_std": 0.03,
    "min_pass_ratio": 0.8
  },
  "unet": {
    "pca_dim": 256,
    "unet_base_ch": 384,
    "timesteps": 200,
    "epochs": 300,
    "batch_size": 1024,
    "lr": 0.0003,
    "cfg_dropout": 0.2,
    "cfg_scale": 1.5,
    "sample_steps": 50,
    "use_ddim": true
  },
  "physics_guidance": {
    "enabled": false,
    "weight": 0.0
  },
  "device": "cuda",
  "num_workers": 0
}
```

### 13.2 配置段说明

| 配置段 | 适用方案 | 关键参数 |
|--------|---------|----------|
| `autoencoder` | 全部 | in_channels, latent_channels, latent_spatial, input_size, epochs, batch_size, lr, loss_weights |
| `latent_gate` | 全部 | max_recon_l1, min_latent_std, min_pass_ratio |
| `diffusion` | mlp/dim_guided/edm_guided/pidm_guided | denoiser, pca_dim, mlp_hidden, timesteps, epochs, cfg_dropout, cfg_scale, sample_steps |
| `unet` | pca_unet/dim_unet | pca_dim, unet_base_ch, timesteps, epochs, cfg_*, sample_steps |
| `edm` | edm_guided | sigma_data, sigma_min, sigma_max, p_mean, p_std, rho |
| `pidm` | pidm_guided | c_data, c_residual, warmup_frac, min_thickness_mm, use_eikonal, eikonal_weight, n_correction |
| `condition_validation` | dim_*/edm/pidm | min_std_ratio, min_abs_std |
| `physics_guidance` | 通用(可选) | enabled, weight |

---

## 14. 常用命令速查

### 14.1 数据准备

```bash
# UIUC 翼型预处理 → NPZ
python run.py prepare-airfoil-uiuc --force

# 固定 train/test 划分
python run.py prepare-splits --dataset airfoil_uiuc_sdf --force
python run.py prepare-splits --dataset single --force
python run.py prepare-splits --dataset F404 --force
```

### 14.2 训练

```bash
# 全流程 (AE + 扩散)
python run.py train --scheme pca_unet --dataset airfoil_uiuc_sdf

# 快速冒烟 (15 epoch AE + 20 epoch 扩散)
python run.py train --scheme pca_unet --dataset airfoil_uiuc_sdf --fast

# 分阶段
python run.py train --scheme pca_unet --dataset airfoil_uiuc_sdf --stage ae
python run.py train --scheme pca_unet --dataset airfoil_uiuc_sdf --stage unet \
  --ae-run-dir outputs/pca_unet/airfoil_uiuc_sdf/20260615_110639/autoencoder

# 其他方案训练
python run.py train --scheme mlp --dataset F404
python run.py train --scheme dim_guided --dataset single
python run.py train --scheme dim_unet --dataset F404 --fast
python run.py train --scheme edm_guided --dataset single
python run.py train --scheme pidm_guided --dataset single
```

### 14.3 测试与评估

```bash
# 标准测试 (AE+扩散)
python run.py test --scheme pca_unet \
  --run-dir outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115

# 翼型几何校核 (SDF→轮廓→条件误差)
python run.py eval-airfoil-geometry \
  --run-dir outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115 \
  --candidates-per-condition 32
```

### 14.4 XFOIL 校核

```bash
python scripts/airfoil/run_xfoil_batch.py \
  --xfoil-bin /home/vipuser/tools/xfoil/build/src/xfoil-6.97 \
  --generated-dir outputs/.../geometry_eval_rerank_c32/top_dat \
  --baseline-raw-dir data/airfoil/raw/uiuc \
  --ranked-csv outputs/.../geometry_eval_rerank_c32/rerank_summary.csv \
  --out-dir outputs/.../xfoil_eval_top50_re1e6_m010_a-2_6 \
  --limit 50 --workers 16 --alpha-start -2 --alpha-end 6 --alpha-step 1 \
  --smooth-passes 8 --te-gap 0.001 --te-blend-start 0.90 --timeout 35 --force

# 生成 XFOIL 结果可视化
python scripts/airfoil/plot_xfoil_results.py \
  --xfoil-dir outputs/.../xfoil_eval_top50_re1e6_m010_a-2_6 \
  --top-k 10
```

### 14.5 计时汇总

```bash
# 汇总所有 outputs 下的 timing.json
python run.py collect-timing

# 按方案/数据集筛选
python run.py collect-timing --scheme pca_unet --dataset airfoil_uiuc_sdf
```

### 14.6 测试

```bash
# 运行全部翼型相关测试 (19 个)
python -m pytest tests/test_airfoil_geometry.py \
  tests/test_airfoil_sdf.py \
  tests/test_prepare_airfoil_uiuc.py \
  tests/test_pca_unet_airfoil_dataset.py \
  tests/test_pca_unet_airfoil_smoke.py \
  tests/test_pca_unet_airfoil_diffusion_config.py
```

---

## 15. 指标参考

### 15.1 子午面数据集 (single, fast: 15+20 epoch)

| 方案 | mean_gen_l1 | AE 重建 L1 | 备注 |
|------|-------------|-----------|------|
| MLP | ~0.036 | ~0.008 | 基线 |
| PCA-UNet | ~0.014 | ~0.008 | **UNet 精度最高** |
| dim_guided | ~0.038 | ~0.009 | 含 ConditionVector 校验 |
| edm_guided | ~0.030 | ~0.008 | Heun 采样 |
| pidm_guided | ~0.033 | ~0.009 | mean_physics_residual ~0.049 |

### 15.2 子午面数据集 (F404, full: 80+200 epoch)

| 方案 | mean_gen_l1 | 备注 |
|------|-------------|------|
| MLP | ~0.037 | |
| PCA-UNet | ~0.030 | |
| dim_guided | ~0.043 | 部分条件列为常量 |
| dim_unet | ~0.025 | **UNet + 192 维 PCA** |
| edm_guided | ~0.033 | |
| pidm_guided | ~0.039 | |

### 15.3 翼型数据集 (airfoil_uiuc_sdf, full: 80+200 epoch)

| 阶段 | 指标 | 值 |
|------|------|-----|
| AE | val L1 | **0.00354** |
| AE | 几何 pass ratio | **100%** |
| 扩散 | mean_gen_l1 | **0.0374** |
| 扩散 | best_gen_l1 | **0.0246** |
| 扩散 | pca_dim | 256 |
| 扩散 | 训练耗时 | 28分54秒 |
| XFOIL | 生成收敛率 | 80% (40/50) |
| XFOIL | 生成平均 L/D | 30.89 |
| XFOIL | 最优生成 L/D | 103.41 (goe188) |
| XFOIL | 最优生成 CL | 1.082 (fx74cl6140) |

---

## 16. 添加新数据集/新方案指南

### 16.1 添加新数据集

1. 在 `data/` 下创建数据集目录，编写 `dataset.json`
2. 确保包含：`processed_dir`, `index_file`, `npz_dir`, `condition_columns`, `split` 配置
3. 在每个目标方案的 `config/` 下创建 `{dataset}.json`
4. 如需预处理，在 `scripts/` 下编写预处理脚本
5. 运行 `python run.py prepare-splits --dataset {name} --force`

### 16.2 添加新方案

1. 在 `schemes/` 下创建新目录，复制一套完整代码（独立维护）
2. `pipeline.py` 必须实现：
   - `run_train(dataset, stage, fast, ae_run_dir) -> Path`
   - `run_test(run_dir) -> None`
3. 在 `config/` 下为各数据集添加 JSON 配置
4. 在 `run.py` 的 `SCHEMES` 字典中注册：
   ```python
   SCHEMES = {
       ...
       "my_scheme": "schemes.my_scheme.pipeline",
   }
   ```

---

## 17. Git 提交历史 (airfoil-conditional-diffusion)

```
01121ba  Add XFOIL result visualizations
4155620  Document XFOIL airfoil evaluation
9a6f5f8  Add batch XFOIL airfoil evaluation
94f379b  Add airfoil geometry rerank export
bb805f9  Prepare pca_unet airfoil diffusion stage
262676f  Add pca_unet airfoil dataset smoke tests
a2f530a  Add UIUC airfoil preprocessing pipeline
981eb13  Add airfoil SDF rasterization
8c57981  Add airfoil geometry primitives
103fc54  Fix pca_unet airfoil dataset assumptions
a3b1e97  更新各个方案训练和测试结果
958c697  新增pidm的分析报告
24fb9ab  新增pidm_guided方法
0378396  增加edm_guided方案
732bbee  修改readme,增加dim_unet
e263658  增加基于几何尺寸的unet扩散,增加训练时间统计对比
a27008f  增加训练和测试任务时间统计
bac129b  新增:基于尺寸扩散的方案dim_guided
04f2b69  清理无用的测试代码
799eb51  删除文件 README.en.md
```

---

> 文档生成时间：2026-06-15  
> 根目录：`/home/vipuser/Meri-FID-Gate-Airfoil`  
> 当前分支：`airfoil-conditional-diffusion`  
> 最新 commit：`01121ba`
