# UIUC 翼型条件扩散实验步骤指南

## 0. 当前准备状态

代码层面已经具备开始实验的基础条件：

1. UIUC/Selig `.dat` 解析、翼型归一化、上下表面重采样、五维几何条件计算已实现。
2. 二维 SDF 栅格化已实现，默认输出 `128 x 256`。
3. `airfoil_uiuc_sdf` 数据集配置已加入。
4. `pca_unet` 翼型配置已加入，默认保留 `device: "cuda"`，当前无 GPU 机器会自动回退 CPU。
5. AE 支持矩形输入：`input_size = [128, 256]`，`latent_spatial = [8, 16]`。
6. diffusion 阶段已支持从 AE checkpoint 恢复矩形尺寸。
7. 已有 19 个 airfoil 相关测试覆盖核心链路。

但还不能说“全量实验已经完成准备”：

1. 真实 UIUC 全量预处理还没有在当前会话中执行并检查结果。
2. 当前机器没有 GPU，不适合跑完整 AE/diffusion 训练。
3. 第一轮全量实验前还需要检查 `filter_report.json`、条件分布和 SDF 样本质量。

结论：当前已经准备好开始“数据预处理与小样本验证”；完整训练实验需要转到 GPU/CUDA 机器后执行。

## 1. 实验目录与关键文件

代码与配置：

```text
scripts/airfoil_geometry.py
scripts/airfoil_sdf.py
scripts/prepare_airfoil_uiuc.py
data/airfoil_uiuc_sdf/dataset.json
schemes/pca_unet/config/airfoil_uiuc_sdf.json
docs/airfoil_gpu_runbook.md
```

输入数据：

```text
data/airfoil/raw/uiuc/coord_seligFmt/
```

预处理输出：

```text
data/airfoil_uiuc_sdf/processed/
  airfoil_index.csv
  train_split.csv
  test_split.csv
  condition_stats.json
  split_meta.json
  filter_report.json
  airfoil_samples/
    <sample_id>.npz
```

训练输出：

```text
outputs/pca_unet/airfoil_uiuc_sdf/<timestamp>/
  autoencoder/
  diffusion/
  timing.json
```

## 2. 当前 CPU 机器上的前置检查

先确认 Python 依赖和测试状态：

```bash
/home/vipuser/miniconda3/bin/python - <<'PY'
import importlib.util
for name in ["numpy", "pandas", "torch", "pytest", "matplotlib", "tqdm"]:
    print(name, bool(importlib.util.find_spec(name)))
PY
```

运行 airfoil 相关测试：

```bash
/home/vipuser/miniconda3/bin/python -m pytest \
  tests/test_airfoil_geometry.py \
  tests/test_airfoil_sdf.py \
  tests/test_prepare_airfoil_uiuc.py \
  tests/test_pca_unet_airfoil_dataset.py \
  tests/test_pca_unet_airfoil_smoke.py \
  tests/test_pca_unet_airfoil_diffusion_config.py
```

通过标准：

```text
19 passed
```

编译关键文件：

```bash
/home/vipuser/miniconda3/bin/python -m compileall \
  scripts/airfoil_geometry.py \
  scripts/airfoil_sdf.py \
  scripts/prepare_airfoil_uiuc.py \
  run.py \
  schemes/pca_unet/pipeline.py \
  schemes/pca_unet/train/train_ae.py \
  schemes/pca_unet/train/train_diffusion.py \
  schemes/pca_unet/test.py
```

## 3. 全量 UIUC 数据预处理

在仓库根目录执行：

```bash
python run.py prepare-airfoil-uiuc --force
```

默认输入目录：

```text
data/airfoil/raw/uiuc/coord_seligFmt
```

如果实际 `.dat` 文件在更深层目录，不需要额外指定；脚本会递归搜索 `*.dat`。

预处理完成后检查输出：

```bash
ls -lh data/airfoil_uiuc_sdf/processed
ls data/airfoil_uiuc_sdf/processed/airfoil_samples | head
```

查看过滤摘要：

```bash
python - <<'PY'
import json
from pathlib import Path
path = Path("data/airfoil_uiuc_sdf/processed/filter_report.json")
report = json.loads(path.read_text(encoding="utf-8"))
print(report["summary"])
for item in report["items"][:10]:
    if item["reasons"]:
        print(item)
PY
```

通过标准：

1. `passed` 应大于 1000。
2. `failed` 样本有明确 `reasons`。
3. `airfoil_samples/*.npz` 数量应等于通过样本数。
4. 随机打开 `.npz` 时包含：
   - `sdf2d_norm`
   - `semantic_mask`
   - `coords_raw`
   - `coords_resampled`
   - `condition_raw`
   - `condition_json`

快速检查 `.npz`：

```bash
python - <<'PY'
from pathlib import Path
import numpy as np

sample = next((Path("data/airfoil_uiuc_sdf/processed/airfoil_samples")).glob("*.npz"))
data = np.load(sample, allow_pickle=True)
print(sample)
for key in data.files:
    arr = data[key]
    print(key, getattr(arr, "shape", None), getattr(arr, "dtype", None))
print("sdf range", data["sdf2d_norm"].min(), data["sdf2d_norm"].max())
print("mask unique", np.unique(data["semantic_mask"]))
PY
```

## 4. 固定 Train/Test 划分

执行：

```bash
python run.py prepare-splits --dataset airfoil_uiuc_sdf --force
```

检查：

```bash
python - <<'PY'
import json
import pandas as pd
from pathlib import Path

root = Path("data/airfoil_uiuc_sdf/processed")
train = pd.read_csv(root / "train_split.csv")
test = pd.read_csv(root / "test_split.csv")
meta = json.loads((root / "split_meta.json").read_text(encoding="utf-8"))
stats = json.loads((root / "condition_stats.json").read_text(encoding="utf-8"))
print("train", len(train), "test", len(test))
print("split_meta", meta)
print("condition columns", stats["columns"])
PY
```

通过标准：

1. `split_meta.requested_test_size == 0.15`。
2. `test_count` 约为总通过样本数的 15%。
3. `condition_stats.columns` 为：

```text
t_max
x_tmax
camber_max
x_camber_max
te_gap
```

## 5. 预训练前数据质量检查

检查条件分布：

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("data/airfoil_uiuc_sdf/processed/train_split.csv")
cols = ["t_max", "x_tmax", "camber_max", "x_camber_max", "te_gap"]
print(df[cols].describe())
PY
```

重点看：

1. `t_max` 是否大多在 `0.02` 到 `0.35`。
2. `x_tmax` 是否在 `[0, 1]`。
3. `camber_max` 是否非负且没有异常长尾。
4. `te_gap` 是否接近非负。

抽查 SDF 符号：

```bash
python - <<'PY'
from pathlib import Path
import numpy as np

paths = list(Path("data/airfoil_uiuc_sdf/processed/airfoil_samples").glob("*.npz"))[:20]
for path in paths:
    data = np.load(path, allow_pickle=True)
    sdf = data["sdf2d_norm"]
    mask = data["semantic_mask"]
    print(path.name, sdf.shape, float(sdf.min()), float(sdf.max()), sorted(np.unique(mask).tolist()))
PY
```

通过标准：

1. `sdf2d_norm.shape == (128, 256)`。
2. SDF 最小值小于 0，最大值大于 0。
3. `semantic_mask` 只包含 `0` 和 `1`。

## 6. 当前 CPU 机器上的最小训练检查

当前机器没有 GPU，不建议跑全量训练。

可以做的检查：

```bash
python run.py train --scheme pca_unet --dataset airfoil_uiuc_sdf --stage ae --fast
```

注意：即使 `--fast`，如果全量 UIUC 数据很大，CPU 仍可能很慢。更推荐只跑测试中的 smoke test：

```bash
/home/vipuser/miniconda3/bin/python -m pytest tests/test_pca_unet_airfoil_smoke.py
```

通过标准：

1. AE 能构造 `input_size = [128, 256]`。
2. `latent_spatial = [8, 16]` 校验通过。
3. CPU 单步 forward/backward 正常。

## 7. GPU 机器上的完整 AE 训练

先确认 CUDA：

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda", torch.version.cuda)
    print("device", torch.cuda.get_device_name(0))
    print("memory_gb", torch.cuda.get_device_properties(0).total_memory / 1024**3)
PY
```

完整 AE 训练：

```bash
python run.py train --scheme pca_unet --dataset airfoil_uiuc_sdf --stage ae
```

产物目录形如：

```text
outputs/pca_unet/airfoil_uiuc_sdf/<timestamp>/autoencoder
```

训练期间记录：

1. GPU 型号。
2. CUDA/PyTorch 版本。
3. batch size。
4. 峰值显存。
5. 每 epoch 时间。
6. 最优 validation L1。
7. `latent_gate_report.json` 是否通过。

如果显存不足，先改：

```json
"autoencoder": {
  "batch_size": 8
}
```

## 8. GPU 机器上的扩散训练

使用 AE 训练输出目录：

```bash
python run.py train \
  --scheme pca_unet \
  --dataset airfoil_uiuc_sdf \
  --stage unet \
  --ae-run-dir outputs/pca_unet/airfoil_uiuc_sdf/<timestamp>/autoencoder
```

训练期间记录：

1. `pca_dim`。
2. `unet_base_ch`。
3. batch size。
4. 峰值显存。
5. `val_gen_l1`。
6. `generation_metrics.json` 中的 `mean_l1_vs_original`。

如果显存不足，先改：

```json
"unet": {
  "batch_size": 16,
  "unet_base_ch": 48
}
```

## 9. 评估与可视化

完整训练后执行：

```bash
python run.py test --scheme pca_unet --run-dir outputs/pca_unet/airfoil_uiuc_sdf/<timestamp>
```

重点检查：

1. `outputs/.../diffusion/generation_metrics.json`
2. generation grid 图片
3. 单样本 SDF 对比图
4. 是否有空白、塌缩或明显破损翼型

第一轮评价标准：

1. 生成 SDF 不为空。
2. 大部分样本内部/外部符号合理。
3. `mean_l1_vs_original` 低于明显随机生成基线。
4. 条件下生成形状有可见厚度/弯度差异。

## 10. 常见失败与处理

### 10.1 `prepare-splits` 找不到 dataset

检查：

```bash
ls data/airfoil_uiuc_sdf/dataset.json
```

### 10.2 训练时报 SDF 尺寸不匹配

检查 `.npz`：

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
path = next(Path("data/airfoil_uiuc_sdf/processed/airfoil_samples").glob("*.npz"))
print(np.load(path)["sdf2d_norm"].shape)
PY
```

配置必须匹配：

```json
"input_size": [128, 256],
"latent_spatial": [8, 16]
```

### 10.3 条件列缺失

检查：

```bash
head -1 data/airfoil_uiuc_sdf/processed/airfoil_index.csv
cat data/airfoil_uiuc_sdf/processed/condition_stats.json
```

必须包含：

```text
t_max,x_tmax,camber_max,x_camber_max,te_gap
```

### 10.4 GPU 不可用

检查：

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
PY
```

如果为 `False`，训练会自动回退 CPU，但完整训练不建议在 CPU 上跑。

## 11. 第一轮实验通过门槛

数据阶段：

1. 通过样本数大于 1000。
2. 条件分布合理。
3. 随机 SDF 抽查无明显符号反转。

AE 阶段：

1. validation L1 稳定下降。
2. 重建图不塌缩。
3. `latent_gate_report.json` 没有明显潜空间异常。

扩散阶段：

1. train loss 和 val loss 没有 NaN。
2. 生成样本不是空白 SDF。
3. 不同条件下厚度/弯度有可见响应。
4. `generation_metrics.json` 可正常生成。

## 12. 建议执行顺序

当前 CPU 机器：

```bash
/home/vipuser/miniconda3/bin/python -m pytest \
  tests/test_airfoil_geometry.py \
  tests/test_airfoil_sdf.py \
  tests/test_prepare_airfoil_uiuc.py \
  tests/test_pca_unet_airfoil_dataset.py \
  tests/test_pca_unet_airfoil_smoke.py \
  tests/test_pca_unet_airfoil_diffusion_config.py

python run.py prepare-airfoil-uiuc --force
python run.py prepare-splits --dataset airfoil_uiuc_sdf --force
```

GPU 机器：

```bash
python run.py train --scheme pca_unet --dataset airfoil_uiuc_sdf --stage ae
python run.py train --scheme pca_unet --dataset airfoil_uiuc_sdf --stage unet --ae-run-dir outputs/pca_unet/airfoil_uiuc_sdf/<timestamp>/autoencoder
python run.py test --scheme pca_unet --run-dir outputs/pca_unet/airfoil_uiuc_sdf/<timestamp>
```
