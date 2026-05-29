# Meridian SDF 尺寸引导潜空间扩散

基于 `sdf2d` 子午面数据的**两阶段生成管线**：自编码器学习潜表示，再在潜空间上训练**尺寸条件扩散模型**。

## 架构概览

| 阶段 | 说明 |
|------|------|
| 1. 自编码器 | SDF + 语义 mask → 潜向量 `z_m` → 重建 SDF |
| 2. 质量门禁 | 校验重建质量与潜空间分布 |
| 3. 条件扩散 | 以 5 维设计参数为条件生成 `z_m`，再解码为 SDF |

**条件向量**：`hub_r_end_mm`、`rim_r_start_mm`、`angle_web_deg`、`r_trans_bore_web_mm`、`z_min`

## 方案与目录

每个扩散方案代码**完全独立**，位于 `schemes/` 下，配置放在各自 `config/` 目录：

| 方案 | `--scheme` | 目录 | 去噪器 |
|------|------------|------|--------|
| MLP 扩散 | `mlp` | `schemes/mlp/` | PCA(128) + MLP |
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

## 参考指标（single 数据集）

| 方案 | mean_gen_l1 | AE 重建 L1 |
|------|-------------|------------|
| MLP | ~0.036 | ~0.008 |
| PCA-UNet | ~0.014 | ~0.008 |

## 许可证

内部研究项目，按需补充许可证信息。
