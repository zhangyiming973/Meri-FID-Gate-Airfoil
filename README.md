# Meridian SDF 尺寸引导潜空间扩散

基于 `sdf2d` 子午面数据的**两阶段生成管线**：自编码器学习潜表示，再在潜空间上训练**尺寸条件扩散模型**。项目提供 **6 套独立方案**（`mlp` / `pca_unet` / `dim_guided` / `dim_unet` / `edm_guided` / `pidm_guided`），统一由 `run.py` 调度。

## 架构概览


| 阶段      | 说明                                 |
| ------- | ---------------------------------- |
| 1. 自编码器 | SDF + 语义 mask → 潜向量 `z_m` → 重建 SDF |
| 2. 质量门禁 | 校验重建质量与潜空间分布                       |
| 3. 条件扩散 | 以 5 维设计参数为条件生成 `z_m`，再解码为 SDF      |


**条件向量**：`hub_r_end_mm`、`rim_r_start_mm`、`angle_web_deg`、`r_trans_bore_web_mm`、`z_min`

## 方案与目录

每个扩散方案代码**完全独立**，位于 `schemes/` 下，配置放在各自 `config/` 目录：


| 方案          | `--scheme`    | 目录                     | 去噪器                                                                                           |
| ----------- | ------------- | ---------------------- | --------------------------------------------------------------------------------------------- |
| MLP 扩散      | `mlp`         | `schemes/mlp/`         | PCA(128) + MLP                                                                                |
| PCA-UNet    | `pca_unet`    | `schemes/pca_unet/`    | PCA(128) → 网格 + UNet                                                                          |
| 尺寸引导扩散      | `dim_guided`  | `schemes/dim_guided/`  | PCA(128) + MLP + ConditionVector 校验                                                           |
| 尺寸 UNet 扩散  | `dim_unet`    | `schemes/dim_unet/`    | PCA(192) + UNet + ConditionVector 校验                                                          |
| EDM 扩散引导    | `edm_guided`  | `schemes/edm_guided/`  | PCA(128) + EDM 预条件化 MLP + Heun 采样                                                             |
| PIDM 物理信息扩散 | `pidm_guided` | `schemes/pidm_guided/` | PCA(128) + MLP + 几何残差虚拟似然（[PIDM](https://github.com/jhbastek/PhysicsInformedDiffusionModels)） |

### 方案速查

| 需求 | 推荐方案 | 扩散 `--stage` |
| ---- | -------- | -------------- |
| 基线 MLP 潜空间扩散 | `mlp` | `diff` |
| 空间 UNet 去噪（更高精度） | `pca_unet` | `unet` |
| Excel/NPZ 五维条件 + 适用性校验 | `dim_guided` | `diff` |
| 上述条件 + UNet 去噪 | `dim_unet` | `unet` |
| EDM 预条件化 + Heun 采样 | `edm_guided` | `diff` |
| PIDM 几何残差虚拟似然 | `pidm_guided` | `diff` |

> MLP 类方案（`mlp` / `dim_guided` / `edm_guided` / `pidm_guided`）传 `--stage unet` 时会自动归一化为 `diff`。


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

# 3. 训练 — dim_guided 方案（Excel/NPZ ConditionVector + 适用性校验）
python run.py train --scheme dim_guided --dataset single
python run.py train --scheme dim_guided --dataset F404 --fast

# 3. 训练 — dim_unet 方案（ConditionVector + UNet 尺寸条件扩散）
python run.py train --scheme dim_unet --dataset single
python run.py train --scheme dim_unet --dataset F404 --fast

# 4. 分阶段训练
python run.py train --scheme mlp --dataset F404 --stage ae
python run.py train --scheme mlp --dataset F404 --stage diff \
  --ae-run-dir outputs/mlp/F404/{timestamp}/autoencoder

python run.py train --scheme pca_unet --dataset F404 --stage unet \
  --ae-run-dir outputs/pca_unet/F404/{timestamp}/autoencoder

python run.py train --scheme dim_guided --dataset single --stage diff \
  --ae-run-dir outputs/dim_guided/single/{timestamp}/autoencoder

python run.py train --scheme dim_unet --dataset single --stage unet \
  --ae-run-dir outputs/dim_unet/single/{timestamp}/autoencoder

# 4b. 训练 — pidm_guided 方案（PIDM 风格几何残差虚拟似然）
python run.py train --scheme pidm_guided --dataset single
python run.py train --scheme pidm_guided --dataset F404 --fast

python run.py train --scheme pidm_guided --dataset single --stage diff \
  --ae-run-dir outputs/pidm_guided/single/{timestamp}/autoencoder

python run.py train --scheme edm_guided --dataset single --stage diff \
  --ae-run-dir outputs/edm_guided/single/{timestamp}/autoencoder

# 5. 快速冒烟
python run.py train --scheme mlp --dataset F404 --fast
python run.py train --scheme edm_guided --dataset single --fast
python run.py train --scheme pidm_guided --dataset single --fast

# 6. 测试 / 可视化（聚合 AE + 扩散产物，替代旧 visualize.py）
python run.py test --scheme mlp --run-dir outputs/mlp/F404/{timestamp}
python run.py test --scheme pca_unet --run-dir outputs/pca_unet/F404/{timestamp}
python run.py test --scheme dim_guided --run-dir outputs/dim_guided/single/{timestamp}
python run.py test --scheme dim_unet --run-dir outputs/dim_unet/single/{timestamp}
python run.py test --scheme edm_guided --run-dir outputs/edm_guided/single/{timestamp}
python run.py test --scheme pidm_guided --run-dir outputs/pidm_guided/single/{timestamp}
```

配置文件自动从 `schemes/{scheme}/config/{dataset}.json` 读取，无需手动指定 `--config`。

### 训练 / 测试计时

各方案训练与测试会自动记录耗时，写入每次 run 目录下的 `timing.json`，并在控制台打印摘要。

**单次 run 的 timing.json 结构示例**：

```json
{
  "scheme": "dim_guided",
  "dataset": "single",
  "run_dir": "outputs/dim_guided/single/20260530_170454",
  "train": {
    "stage_requested": "all",
    "fast": false,
    "started_at": "2026-05-30T17:04:54+08:00",
    "finished_at": "2026-05-30T18:30:12+08:00",
    "total_sec": 5118.5,
    "total_human": "1h 25m 18.5s",
    "stages": {
      "condition_validation": {"sec": 0.12, "human": "0.1s"},
      "autoencoder": {"sec": 2100.3, "human": "35m 0.3s"},
      "diffusion": {"sec": 3018.0, "human": "50m 18.0s"}
    }
  },
  "test": {
    "total_sec": 45.2,
    "total_human": "45.2s",
    "stages": {
      "test_eval": {"sec": 45.2, "human": "45.2s"}
    }
  }
}
```


| 输出位置                                           | 说明                    |
| ---------------------------------------------- | --------------------- |
| `{run_dir}/timing.json`                        | 训练 + 测试完整计时           |
| `{run_dir}/visualizations/timing.json`         | 测试阶段复制一份，便于与报告一起归档    |
| `{run_dir}/visualizations/summary_report.json` | 含 `timing` 字段，指标与计时合一 |


**汇总所有 run（汇报用 CSV）**：

```bash
# 汇总 outputs/ 下全部 timing.json
python run.py collect-timing

# 按方案 / 数据集筛选
python run.py collect-timing --scheme mlp --dataset single
python run.py collect-timing --scheme dim_guided --dataset F404
python run.py collect-timing --scheme dim_unet --dataset single
python run.py collect-timing --scheme edm_guided --dataset single
python run.py collect-timing --scheme pidm_guided --dataset single
```

生成文件：

- `outputs/timing_summary.json` — 全部 run 的 JSON 列表
- `outputs/timing_summary.csv` — 表格（scheme、dataset、train/test 耗时、AE/扩散分阶段秒数）
- `outputs/timing_comparison.png` — 各方案各阶段耗时对比图（同一 scheme/dataset 取最新 run）

### `test` 子命令输出

`run.py test` 从 `autoencoder/` 与 `diffusion/` 读取训练阶段产物，汇总到 run 根目录 `visualizations/`：


| 文件                            | 说明                                |
| ----------------------------- | --------------------------------- |
| `summary_report.json`         | AE 门控、扩散 L1 指标、计时、已复制 artifact 列表 |
| `ae_training_dashboard.png` 等 | 从 AE 阶段复制的训练曲线与重建图                |
| `diff_grid_test.png` 等        | 从扩散阶段复制的生成网格与指标图                  |
| `test_generation_l1.png`      | 逐样本生成 L1 vs 原始 SDF                |
| `test_ae_recon_l1.png`        | 逐样本 AE 重建 L1 vs 原始 SDF            |
| `test_gen_vs_ae_recon.png`    | 生成 vs AE 重建并排对比                   |
| `generations/`                | 各测试样本 SDF 三联图副本                   |


`pidm_guided` 方案的 `summary_report.json` 额外包含 `mean_physics_residual`（几何残差均值）与 `pidm_config.json` 副本。

## 输出目录

统一格式：`outputs/{方案}/{数据集}/{时间戳}/`

```
outputs/mlp/F404/20260529_234932/
├── logs/train.log
├── autoencoder/          # AE 权重、潜向量、Gate 报告、重建可视化
├── diffusion/            # 扩散模型、生成指标、生成可视化
└── visualizations/       # run.py test 汇总报告（训练曲线、对比图、summary_report.json）

outputs/pca_unet/single/20260529_120000/
├── logs/train.log
├── autoencoder/
├── diffusion/
└── visualizations/

outputs/dim_guided/single/20260530_153231/
├── logs/train.log
├── timing.json                 # 训练/测试耗时统计
├── condition_applicability.json   # 五维条件适用性评估报告
├── autoencoder/
├── diffusion/
└── visualizations/

outputs/dim_unet/single/20260530_180000/
├── logs/train.log
├── timing.json
├── condition_applicability.json   # 与 dim_guided 相同的五维条件适用性报告
├── autoencoder/                   # 轮缘加权 AE（含 rim / gradient 损失）
├── diffusion/                     # PCA(192) 网格 + 条件 UNet 权重与生成可视化
└── visualizations/

outputs/pidm_guided/single/20260531_124746/
├── logs/train.log
├── timing.json
├── autoencoder/
├── diffusion/
│   ├── pidm_config.json           # PIDM 超参与残差分量说明
│   └── generation_metrics.json    # 含 mean_physics_residual
└── visualizations/
```

## 项目结构

```
meri-fid-gate/
├── run.py                      # 统一入口
├── scripts/
│   ├── prepare_splits.py       # 生成固定 train/test 划分
│   ├── split_utils.py
│   └── timing_utils.py         # 训练/测试计时与 collect-timing 汇总
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
│   ├── pca_unet/               # PCA-UNet 方案（独立代码）
│   │   ├── config/
│   │   ├── pipeline.py
│   │   └── ...
│   ├── dim_guided/             # 尺寸引导 MLP 扩散（独立代码）
│   │   ├── config/
│   │   ├── data/condition_vector.py
│   │   ├── pipeline.py
│   │   └── ...
│   ├── dim_unet/               # 尺寸 UNet 扩散（独立代码）
│   │   ├── config/
│   │   ├── data/condition_vector.py
│   │   ├── models/pca_unet.py
│   │   ├── pipeline.py
│   │   └── ...
│   ├── edm_guided/             # EDM 扩散引导（独立代码）
│   │   ├── config/
│   │   ├── models/edm_schedule.py
│   │   ├── pipeline.py
│   │   └── ...
│   └── pidm_guided/            # PIDM 物理信息扩散（独立代码）
│       ├── config/
│       ├── physics/            # 几何残差 + 虚拟似然损失
│       ├── report.md           # 与 PIDM 原版的对比分析
│       ├── models/diffusion.py # 含 posterior_variance 与 x0 校正
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

## dim_guided 方案（尺寸引导扩散）

`dim_guided` 在潜空间条件扩散基础上，增加了 **ConditionVector** 的统一读取与适用性校验，支持从 Excel、NPZ、CSV 索引多级回退加载五维尺寸参数。

### 五维 ConditionVector


| 字段                    | 含义            |
| --------------------- | ------------- |
| `hub_r_end_mm`        | 轮毂外径半径 (mm)   |
| `rim_r_start_mm`      | 轮缘内径半径 (mm)   |
| `angle_web_deg`       | 腹板倾角 (°)      |
| `r_trans_bore_web_mm` | 孔-腹板过渡半径 (mm) |
| `z_min`               | 轴向下界 (mm)     |


### 条件加载优先级

1. `train_split.csv` / `test_split.csv` 索引行中的条件列
2. 内存参数表（CSV 或 Excel）
3. `dataset.json` 中配置的 Excel/CSV 参数表路径
4. NPZ 内 `condition_json`，或从 `engineering_curves_json` 推导

**Excel 配置示例**（在 `data/{dataset}/dataset.json` 中添加）：

```json
{
  "param_table_path": "data/my_dataset/processed/param_table.csv",
  "condition_excel_path": "data/my_dataset/processed/params.xlsx"
}
```

Excel / CSV 参数表需包含 `sample_id`（或 `id`）列，以及五维条件列。支持列名别名，例如 `r_bore2_mm` → `hub_r_end_mm`、`r_outer2_mm` → `rim_r_start_mm`。

### 编程接口

```python
from schemes.dim_guided.data.condition_vector import (
    load_from_excel,
    load_from_npz,
    load_condition_vector,
    evaluate_applicability,
)

# 从 Excel 读取单样本
cv = load_from_excel("params.xlsx", sample_id="sample_001")
print(cv.as_dict())  # {'hub_r_end_mm': 122.03, ...}

# 从 NPZ 读取（需含 condition_json 或可推导的 engineering_curves_json）
cv = load_from_npz("sample_001.npz")

# 多级回退加载
cv = load_condition_vector("sample_001", index_row=row, npz_path=Path("sample_001.npz"))
```

### 适用性校验

训练开始前自动评估条件列是否适合作为扩散条件，结果写入 `condition_applicability.json`：

- 各列缺失率
- 各列标准差与取值范围
- 低方差列（近常量，CFG 区分度不足）
- 物理约束违反样本数

**数据集参考结论**：


| 数据集      | 适用性  | 说明                                                                                     |
| -------- | ---- | -------------------------------------------------------------------------------------- |
| `single` | 适用   | 五维均有足够方差                                                                               |
| `F404`   | 部分适用 | `hub_r_end_mm`、`r_trans_bore_web_mm` 为常量；主要依赖 `rim_r_start_mm`、`angle_web_deg`、`z_min` |


F404 的 NPZ 文件当前不含 `condition_json`，需通过索引 CSV 或 Excel 参数表提供条件。

### 使用命令

```bash
# 全流程训练（含适用性校验 → AE → 扩散）
python run.py train --scheme dim_guided --dataset single

# 分阶段
python run.py train --scheme dim_guided --dataset single --stage ae
python run.py train --scheme dim_guided --dataset single --stage diff \
  --ae-run-dir outputs/dim_guided/single/{timestamp}/autoencoder

# 快速冒烟（15 epoch AE + 20 epoch 扩散）
python run.py train --scheme dim_guided --dataset single --fast

# 测试 / 可视化
python run.py test --scheme dim_guided --run-dir outputs/dim_guided/single/{timestamp}
```

配置文件位于 `schemes/dim_guided/config/{dataset}.json`，其中 `condition_validation` 段可调整适用性判定阈值（`min_std_ratio`、`min_abs_std`）。

## dim_unet 方案（尺寸 UNet 扩散）

`dim_unet` 结合 **dim_guided** 的 ConditionVector 数据链路与 **pca_unet** 的条件 UNet 去噪器，在 PCA 网格潜空间上以 5 维尺寸参数为条件做扩散生成。

### 训练流程

1. **条件适用性校验** — 评估五维 ConditionVector 是否适合作为扩散条件，写入 `condition_applicability.json`
2. **自编码器** — 轮缘加权 AE（含 `rim`、`gradient` 损失）将 SDF 编码为 `z_m`
3. **PCA + UNet 扩散** — 将 `z_m` 经 PCA(192) 投影后重塑为 2D 网格，用条件 UNet 学习 DDPM 去噪
4. **采样与解码** — DDIM 采样 → PCA 逆变换 → AE 解码为子午线 SDF

### 与相关方案对比


| 对比项           | pca_unet        | dim_guided                     | dim_unet        |
| ------------- | --------------- | ------------------------------ | --------------- |
| 去噪器           | 条件 UNet（PCA 网格） | MLP                            | 条件 UNet（PCA 网格） |
| 条件输入          | 索引 CSV 五维列      | ConditionVector（Excel/NPZ/CSV） | 同 dim_guided    |
| AE 损失         | 标准 L1/L2/零等值面   | 轮缘加权 + gradient                | 同 dim_guided    |
| 适用性校验         | 无               | 有                              | 有               |
| PCA 维度        | 128             | 192                            | 192             |
| 配置段名          | `unet`          | `diffusion`                    | `unet`          |
| 分阶段 `--stage` | `unet`          | `diff`                         | `unet`          |


### ConditionVector

五维参数、加载优先级、Excel 配置与适用性判定规则与 **dim_guided 完全相同**（见上一节）。`dim_unet` 在 `schemes/dim_unet/data/condition_vector.py` 中独立维护一份相同接口的实现。

编程接口示例：

```python
from schemes.dim_unet.data import CONDITION_COLUMNS
from schemes.dim_unet.data.condition_vector import (
    load_from_excel,
    load_from_npz,
    load_condition_vector,
    evaluate_applicability,
)

cv = load_condition_vector("sample_001", index_row=row, npz_path=Path("sample_001.npz"))
report = evaluate_applicability(train_df, condition_columns=CONDITION_COLUMNS)
```

### 配置说明

配置文件位于 `schemes/dim_unet/config/{dataset}.json`：


| 配置段                    | 说明                                                                      |
| ---------------------- | ----------------------------------------------------------------------- |
| `autoencoder`          | AE 结构与损失权重（含 `rim`、`gradient`）                                          |
| `latent_gate`          | 潜空间质量门控阈值                                                               |
| `unet`                 | 扩散超参：`pca_dim`（192）、`unet_base_ch`、`timesteps`、`sample_steps`（80）、CFG 等 |
| `condition_validation` | 适用性判定阈值（`min_std_ratio`、`min_abs_std`）                                  |
| `physics_guidance`     | 可选物理引导（默认关闭）                                                            |


与 `pca_unet` 的主要差异：`pca_dim=192`（保留更多潜空间变化方向）、`sample_steps=80`，且 AE 使用轮缘加权损失。

### 使用命令

```bash
# 全流程训练（含适用性校验 → AE → UNet 扩散）
python run.py train --scheme dim_unet --dataset single
python run.py train --scheme dim_unet --dataset F404 --fast

# 分阶段（扩散阶段用 --stage unet，与 pca_unet 相同）
python run.py train --scheme dim_unet --dataset single --stage ae
python run.py train --scheme dim_unet --dataset single --stage unet \
  --ae-run-dir outputs/dim_unet/single/{timestamp}/autoencoder

# 测试 / 可视化
python run.py test --scheme dim_unet --run-dir outputs/dim_unet/single/{timestamp}
python run.py test --scheme dim_unet --run-dir outputs/dim_unet/F404/{timestamp}
```

## edm_guided 方案（EDM 扩散引导）

基于 Karras et al. EDM 框架（见 `wenxian/` 文献），在 PCA 潜空间上实现**预条件化扩散 + Classifier-Free Guidance + Heun 二阶采样**。

### 文献方法摘要

| 组件 | 方法 |
| ---- | ---- |
| 去噪器 | EDM 预条件化包装：`D = c_skip·x + c_out·F(c_in·x; c_noise)` |
| 训练噪声 | `ln σ ~ N(P_mean, P_std²)`，默认 `P_mean=-1.2`, `P_std=1.2` |
| 损失 | 预条件化 MSE，`λ(σ) = 1/c_out(σ)²` |
| 采样 | Heun 二阶 ODE，`ρ=7`，默认 50 步 |
| 引导 | Classifier-Free Guidance |


### 使用命令

```bash
# 全流程训练
python run.py train --scheme edm_guided --dataset single
python run.py train --scheme edm_guided --dataset F404 --fast

# 分阶段
python run.py train --scheme edm_guided --dataset single --stage ae
python run.py train --scheme edm_guided --dataset single --stage diff \
  --ae-run-dir outputs/edm_guided/single/{timestamp}/autoencoder

# 测试 / 可视化
python run.py test --scheme edm_guided --run-dir outputs/edm_guided/single/{timestamp}
```

配置文件位于 `schemes/edm_guided/config/{dataset}.json`，`edm` 段控制 EDM 超参（`sigma_data` 为 null 时从训练集 PCA 编码自动估计）。

## pidm_guided 方案（PIDM 物理信息扩散）

基于 [Physics-Informed Diffusion Models (PIDM, ICLR 2025)](https://github.com/jhbastek/PhysicsInformedDiffusionModels) 的损失设计，在 **PCA 潜空间 MLP 扩散** 上引入**几何残差虚拟似然**，使生成样本趋近独立物理约束（而非对齐 GT 风险代理）。

详细对比分析见 [`schemes/pidm_guided/report.md`](schemes/pidm_guided/report.md)。

### 与 PIDM 原版的对应关系

| PIDM 原版 | `pidm_guided` 适配 |
| --------- | ------------------ |
| Darcy / FEM PDE 残差 → 0 | 壁厚违反 + Eikonal 残差 → 0 |
| 残差虚拟似然（目标 r=0） | 同样形式，方差绑定 `posterior_variance_clipped[t]` |
| 场空间 64×64 直接扩散 | AE 潜空间 PCA(128) + MLP |
| 推理 N/M 步 x₀ 校正 | 可选 `n_correction` / `m_correction` |

### 训练损失

```
L = c_data * L_DDPM + c_residual * (-log p(r=0 | x0_pred, var_t))
```

- 从预测噪声反推 `x0_hat`，经 PCA 逆变换 + AE 解码得到 SDF
- 在 SDF 上计算几何残差（目标为 0，非对齐 GT）
- 前 `warmup_frac`（默认 33%）epoch 仅训练 DDPM，之后启用物理项

**残差分量**（见 `schemes/pidm_guided/physics/geometry_residual.py`）：


| 分量                    | 说明                                                          |
| --------------------- | ----------------------------------------------------------- |
| `thickness_violation` | `ReLU(min_thickness_mm - 估计壁厚)`                             |
| `eikonal`             | SDF Eikonal 条件 `\|\nabla SDF\| ≈ 1`（有限差分，权重 `eikonal_weight`） |
| `area_deviation`      | 可选，与 NPZ `physics.area_mm2` 对齐（`use_area_constraint: true`） |


### 与 `mlp` 方案对比


| 对比项  | `mlp`                                | `pidm_guided`                    |
| ---- | ------------------------------------ | -------------------------------- |
| 去噪器  | PCA(128) + MLP                       | 同左                               |
| 扩散框架 | DDPM + DDIM + CFG                    | 同左                               |
| 物理引导 | `physics_guidance`（默认关，MSE 对齐 GT 风险） | PIDM 虚拟似然（默认开，残差 → 0）            |
| 推理校正 | 无                                    | 可选 x₀ 梯度校正                       |
| 评估指标 | gen L1                               | gen L1 + `mean_physics_residual` |


### 配置说明

配置文件位于 `schemes/pidm_guided/config/{dataset}.json`：


| 配置段           | 说明                                    |
| ------------- | ------------------------------------- |
| `autoencoder` | 与 `mlp` 相同                            |
| `latent_gate` | 潜空间质量门控                               |
| `diffusion`   | MLP 扩散超参（`pca_dim`、`timesteps`、CFG 等） |
| `pidm`        | PIDM 专用超参（见下表）                        |


**`pidm` 段主要参数**：


| 参数                     | 默认值   | 说明                     |
| ---------------------- | ----- | ---------------------- |
| `c_data`               | 1.0   | DDPM 噪声损失权重            |
| `c_residual`           | 0.001 | 残差虚拟似然权重（0 则退化为纯 DDPM） |
| `warmup_frac`          | 0.33  | 前若干 epoch 仅训 DDPM      |
| `min_thickness_mm`     | 2.0   | 最小壁厚约束 (mm)            |
| `use_eikonal`          | true  | 是否启用 Eikonal 残差        |
| `eikonal_weight`       | 0.1   | Eikonal 残差缩放           |
| `use_area_constraint`  | false | 是否约束截面积                |
| `n_correction`         | 0     | DDIM 采样每步 x₀ 梯度校正次数    |
| `m_correction`         | 0     | 末步 x₀ 梯度校正次数           |
| `correction_step_size` | 0.05  | 校正步长                   |


### 使用命令

```bash
# 全流程训练
python run.py train --scheme pidm_guided --dataset single
python run.py train --scheme pidm_guided --dataset F404 --fast

# 分阶段（扩散阶段用 --stage diff，与 mlp 相同）
python run.py train --scheme pidm_guided --dataset single --stage ae
python run.py train --scheme pidm_guided --dataset single --stage diff \
  --ae-run-dir outputs/pidm_guided/single/{timestamp}/autoencoder

# 测试 / 可视化（含 physics_residual 指标）
python run.py test --scheme pidm_guided --run-dir outputs/pidm_guided/single/{timestamp}
```

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

用于 `pca_unet` 与 `dim_unet` 方案。`pca_unet` 默认 `pca_dim=128`；`dim_unet` 使用 `pca_dim=192`，保留更多潜空间主成分。

**架构设计**：

- **PCA → 网格化**：k 维 PCA 系数自动分解为 C×H×W 网格（优先接近正方形、H/W 为 4 的倍数以适配两次 stride=2 下采样）
- **Encoder-Decoder UNet**：
  - Encoder：3层下采样（Conv + GroupNorm + SiLU）
  - Decoder：3层上采样（Upsample + Conv + GroupNorm + SiLU）
  - Skip Connection：保留空间细节
- **条件注入**：5 维工况向量（`dim_unet` 来自 ConditionVector）广播到空间维度后与噪声 latent 拼接

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
  - `mlp` / `dim_guided` / `dim_unet` 等：`physics_guidance` 对齐 GT 风险代理（默认关闭）
  - `pidm_guided`：PIDM 虚拟似然，几何残差目标为 0（默认 `c_residual=0.001`）
  - 详见 [`schemes/pidm_guided/report.md`](schemes/pidm_guided/report.md) 与 `schemes/pidm_guided/physics/`

## 数据流详解

### 输入数据格式

**NPZ 文件内容**：

- `sdf`：256×256 符号距离场（负值表示实体内部）
- `semantic`：语义 mask（hub/web/rim 区域标记）
- `condition_json`：5 维工况向量（可选）
- `physics`：物理属性（最小壁厚、截面面积）

**条件向量来源**：

1. `meridian_index.csv` / `train_split.csv`：直接包含 5 维工况列
2. `param_table.csv` 或 `.xlsx`：参数表补全缺失列（`dim_guided` / `dim_unet` 方案支持 Excel）
3. NPZ 内 `condition_json` 或 `engineering_curves_json` 推导（`dim_guided` / `dim_unet` 方案）
4. `condition_specs.jsonl`：从几何规格推导工况

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

- 将离散时间步 t ∈ [0, T] 映射为连续向量
- 不同频率捕捉不同时间尺度特征
- Transformer 经典做法，适用于扩散模型

### 2. Classifier-Free Guidance (CFG)

**训练阶段**：

- 随机 dropout 条件向量（概率 `cfg_dropout=0.2`）
- 学习条件与无条件两种模式

**推理阶段**：

- 同时计算条件与无条件预测
- 按 `cfg_scale`（默认 1.5）加权混合
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


### UNet 参数（pca_unet / dim_unet 方案）


| 参数             | pca_unet 默认 | dim_unet 默认 | 说明         |
| -------------- | ----------- | ----------- | ---------- |
| `pca_dim`      | 128         | 192         | PCA 压缩维度   |
| `unet_base_ch` | 64          | 64          | UNet 基础通道数 |
| `sample_steps` | 50          | 80          | DDIM 采样步数  |
| `time_dim`     | 128         | 128         | 时间嵌入维度     |


### PIDM 参数（pidm_guided 方案）


| 参数                 | 默认值   | 说明            |
| ------------------ | ----- | ------------- |
| `c_data`           | 1.0   | DDPM 损失权重     |
| `c_residual`       | 0.001 | 残差虚拟似然权重      |
| `warmup_frac`      | 0.33  | 物理项 warmup 比例 |
| `min_thickness_mm` | 2.0   | 最小壁厚约束        |
| `use_eikonal`      | true  | Eikonal 残差开关  |
| `n_correction`     | 0     | 采样 x₀ 校正步数    |


## 参考指标

以下为 `outputs/` 中近期 full/fast 训练 run 的 `mean_gen_l1`（越低越好）。`pidm_guided` 另报告 `mean_physics_residual`。

### single 数据集（fast，~15 epoch AE + 20 epoch 扩散）

| 方案 | mean_gen_l1 | AE 重建 L1 | 备注 |
| ---- | ----------- | ---------- | ---- |
| MLP | ~0.036 | ~0.008 | 基线 |
| PCA-UNet | ~0.014 | ~0.008 | UNet 精度最高 |
| dim_guided | ~0.038 | ~0.009 | 含 ConditionVector 校验 |
| edm_guided | ~0.030 | ~0.008 | Heun 采样 |
| pidm_guided | ~0.033 | ~0.009 | `mean_physics_residual` ~0.049 |

### F404 数据集（full，80 epoch AE + 200 epoch 扩散）

| 方案 | mean_gen_l1 | 备注 |
| ---- | ----------- | ---- |
| MLP | ~0.037 | |
| PCA-UNet | ~0.030 | |
| dim_guided | ~0.043 | 部分条件列为常量 |
| dim_unet | ~0.025 | UNet + 192 维 PCA |
| edm_guided | ~0.033 | |
| pidm_guided | ~0.039 | `mean_physics_residual` 见 run 目录 |

> 指标随 run 变化，以各 run 下 `diffusion/summary.json` 为准。汇总耗时：`python run.py collect-timing`。


## 翼型到三维机翼

`scripts/generate_wing.py` 可将 Selig/UIUC 格式翼型坐标生成一侧梯形机翼。脚本采用 CadQuery 作为 OpenCascade 封装导出 STEP；未安装 CadQuery 时可用 `--no-step` 先输出几何参数 metadata。

默认三翼型示例使用本地 UIUC/Selig 原始坐标库中的 `n0012`、`naca2412`、`s1223`。这些坐标比生成结果目录里的 `top_dat` 更适合 CAD loft；脚本还会先拆分上下表面并用 cosine 网格重采样，避免原始点列中的局部尖峰造成截面破损。图片反推轮廓可作为无坐标数据时的兜底，但需要尺度标定和轮廓数字化，不建议作为首选几何来源。

参数约定：

| 参数 | 说明 |
| ---- | ---- |
| `--root-tip-ratio` | 根梢比，定义为 `root_chord / tip_chord` |
| `--taper-ratio` | 梢根比，定义为 `tip_chord / root_chord`，与 `--root-tip-ratio` 二选一 |
| `--aspect-ratio` | 常规定义的全翼展弦比，`full_span^2 / full_area` |
| `--total-length` | 单侧展长，上反角作用前的长度 |
| `--dihedral` | 上反角，单位为度 |

示例：对已选的三个生成翼型输出 metadata：

```bash
python scripts/generate_wing.py \
  --example-three \
  --root-tip-ratio 2.0 \
  --aspect-ratio 8.0 \
  --total-length 5.0 \
  --dihedral 5.0 \
  --no-step
```

安装 CadQuery 后导出 STEP：

```bash
python scripts/generate_wing.py \
  --airfoil data/airfoil/raw/uiuc/coord_seligFmt/coord_seligFmt/naca2412.dat \
  --root-tip-ratio 2.0 \
  --aspect-ratio 8.0 \
  --total-length 5.0 \
  --dihedral 5.0 \
  --output outputs/wing_3d_clean/naca2412_wing.step
```
