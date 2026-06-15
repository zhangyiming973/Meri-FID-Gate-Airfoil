# Meri-FID-Gate-Airfoil 翼型生成、校核与三维重建实验总结

日期：2026-06-15  
项目目录：`/home/vipuser/Meri-FID-Gate-Airfoil`

## 1. 实验目标

本轮实验围绕翼型生成结果的工程可用性展开，目标不是只看扩散模型的 SDF 误差，而是把生成结果走完整链路：

1. UIUC/Selig 翼型数据预处理与几何条件提取。
2. PCA-UNet 条件扩散生成翼型 SDF。
3. 从生成 SDF 提取光顺翼型 `.dat`。
4. 用 XFOIL 做气动校核。
5. 用 CadQuery/OpenCascade 生成三维机翼 STEP。

本轮关键改进是修正了 `SDF -> 翼型坐标` 的采样路径。旧方法从 `matplotlib.contour` 顶点顺序直接拆上下表面，容易把 contour 顶点跳变放大成翼型突变和尖峰。新方法改为在固定 `x/c` 网格上直接沿 SDF 竖线寻找上下零交点，使 raw generated 翼型本身更平滑，后续 XFOIL 和 CadQuery 的结果明显改善。

## 2. 数据与几何处理模块

核心文件：

```text
scripts/airfoil_geometry.py
scripts/airfoil_sdf.py
scripts/prepare_airfoil_uiuc.py
```

### 2.1 Selig/UIUC `.dat` 解析

`scripts/airfoil_geometry.py` 提供基础几何函数：

- `parse_selig_dat(path)`：读取 UIUC/Selig 坐标文件，跳过标题和非数值行。
- `normalize_airfoil(coords)`：归一化弦长到 `[0, 1]`，并做线性弦线校正。
- `split_surfaces(coords)`：按前缘索引拆分上下表面。
- `cosine_x_grid(n)`：生成前缘/尾缘加密的 cosine x 网格。
- `resample_surfaces(surfaces, x_grid)`：把上下表面插值到统一 x 网格。
- `compute_airfoil_conditions(resampled)`：计算五维条件：
  - `t_max`
  - `x_tmax`
  - `camber_max`
  - `x_camber_max`
  - `te_gap`
- `validate_airfoil(resampled, conditions)`：基础几何质量过滤。

### 2.2 SDF 数据集构建

`scripts/airfoil_sdf.py` 将重采样翼型转为二维 SDF 和语义 mask。

`scripts/prepare_airfoil_uiuc.py` 批量处理 UIUC 原始翼型：

```bash
/home/vipuser/miniconda3/bin/python run.py prepare-airfoil-uiuc --force
```

主要输出：

```text
data/airfoil_uiuc_sdf/processed/airfoil_index.csv
data/airfoil_uiuc_sdf/processed/filter_report.json
data/airfoil_uiuc_sdf/processed/condition_stats.json
data/airfoil_uiuc_sdf/processed/airfoil_samples/*.npz
```

每个 `.npz` 中包含：

- `sdf2d_norm`
- `semantic_mask`
- `coords_raw`
- `coords_resampled`
- 五维几何条件

## 3. PCA-UNet 条件扩散生成路径

核心文件：

```text
schemes/pca_unet/train/train_ae.py
schemes/pca_unet/train/train_diffusion.py
schemes/pca_unet/models/autoencoder.py
schemes/pca_unet/models/diffusion.py
schemes/pca_unet/models/latent_pca.py
schemes/pca_unet/train/diffusion_codec.py
```

### 3.1 Autoencoder 阶段

AE 将二维 SDF 压缩到 latent 表示，并可从 latent 解码回 SDF。

典型调用：

```bash
/home/vipuser/miniconda3/bin/python run.py train \
  --scheme pca_unet \
  --dataset airfoil_uiuc_sdf \
  --stage ae
```

本次使用的 AE 输出目录：

```text
outputs/pca_unet/airfoil_uiuc_sdf/20260615_110639/autoencoder/
```

### 3.2 PCA-UNet 扩散阶段

扩散阶段先对 AE latent 做 PCA 编码，再训练条件 UNet 在 PCA latent 网格上生成样本。条件输入为五维几何条件归一化向量。

典型调用：

```bash
/home/vipuser/miniconda3/bin/python run.py train \
  --scheme pca_unet \
  --dataset airfoil_uiuc_sdf \
  --stage unet \
  --ae-run-dir outputs/pca_unet/airfoil_uiuc_sdf/20260615_110639/autoencoder
```

本次扩散输出目录：

```text
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/diffusion/
```

关键文件：

```text
best_unet_diffusion.pt
latent_pca.json
generation_metrics.json
summary.json
generations/*.png
generated_sdf_npz/*.npz
```

本次生成结果摘要：

```text
mean_gen_l1       = 0.03740977035149148
mean_ae_recon_l1  = 0.0035532731289488946
best_gen_l1       = 0.02460429333150387
pca_dim           = 256
denoiser          = pca_unet
latent grid       = 1 x 16 x 16
```

`generated_sdf_npz/*.npz` 是本轮后续几何重建的关键输入。每个文件包含：

```text
generated_sdf
target_sdf
ae_recon_sdf
condition_raw
condition_norm
condition_columns
condition_json
sample_id
source_npz_path
l1_vs_original
l1_ae_recon
```

## 4. 生成 SDF 到翼型 `.dat` 的几何重建

核心文件：

```text
scripts/airfoil/evaluate_generated_airfoils.py
```

### 4.1 旧路径的问题

旧路径：

```text
generated_sdf
  -> matplotlib.contour 零等值线
  -> contour 顶点序列
  -> normalize_airfoil
  -> split_surfaces
  -> resample_surfaces
  -> .dat
```

问题：

- contour 顶点顺序可能发生局部跳段。
- `split_surfaces` 假设闭合点列顺序稳定。
- 局部毛刺和多段近接会在上下表面拆分和线性插值中放大。
- raw generated 翼型出现突变、尖峰、坑洼。
- XFOIL 临时平滑只能缓解，无法从源头修复。
- CadQuery 截面若直接用这类 `.dat`，STEP 截面也不光顺。

### 4.2 新路径：SDF 竖线直接采样

本轮新增/使用的新逻辑：

```text
generated_sdf
  -> contour 只用于估计弦向范围
  -> cosine x/c 网格
  -> 每个 x/c 上沿 y 方向直接找 SDF 零交点
  -> 选取主上下表面
  -> camber/thickness 轻量 despike
  -> ResampledAirfoil
  -> .dat
```

对应函数：

```text
_interp_sdf_column
_zero_crossings
_surface_pair_from_sdf_column
_sample_sdf_airfoil
_despike_line
```

`_candidate_from_sdf` 现在优先使用 `_sample_sdf_airfoil`，失败时才回退旧的 contour 路径。

### 4.3 本轮新几何导出

输入：

```text
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/diffusion/generated_sdf_npz/
```

输出：

```text
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/geometry_eval_generated_sdf_direct/
```

关键产物：

```text
top_dat/*.dat
candidate_dat/{sample_id}/rank01_cand000.dat
rerank_summary.csv
per_sample_geometry.csv
geometry_eval_report.json
```

本次结果：

```text
num_inputs   = 247
num_airfoils = 247
num_valid    = 247
num_failures = 0
```

按条件误差排序的 top32：

```text
n6h10
goe282
ua2-180
ag455ct02r
oa209
e174
e330
ah80129
goe457
jx-gs-10
goe670
rc10n1
goe622
pmc19sm
fx08s176
ah93157
e195
rc08n1
nlr7301
cr001sm
e583
fx60100sm
mh43
jx-gs-04
ncambre
goe269
rg8
k3311
sa7035
ah79k135
sd2030
fx66h80
```

### 4.4 采样质量预览

生成样本 SDF 采样预览：

```text
outputs/generated_sdf_sampling_preview_20260615/
```

包含：

```text
goe188_generated_sdf_sampling.png
m14_generated_sdf_sampling.png
fx74cl6140_generated_sdf_sampling.png
ua2-180_generated_sdf_sampling.png
rc08n1_generated_sdf_sampling.png
```

图中：

- 左上：生成 SDF 热力图和零等值线。
- 右上：旧 contour 顺序采样 vs 新 SDF 竖线采样。
- 左下：厚度线对比。
- 右下：弯度线对比。

新采样使厚度二阶差分 RMS 从约 `0.007~0.036` 降到约 `0.0005~0.0007`。

测试：

```bash
/home/vipuser/miniconda3/bin/python -m pytest tests/test_airfoil_cleanup_rerank.py
```

结果：

```text
4 passed
```

## 5. XFOIL 气动校核模块

核心文件：

```text
scripts/airfoil/run_xfoil_batch.py
scripts/airfoil/plot_xfoil_results.py
```

### 5.1 批量 XFOIL 评估

`run_xfoil_batch.py` 完成：

1. 读取几何 rerank 表。
2. 收集生成翼型 `.dat`。
3. 在 UIUC 原始目录中寻找同名 baseline `.dat`。
4. 对每个 case 写入短路径工作目录。
5. 对送入 XFOIL 的翼型做 XFOIL 专用清洗：
   - cosine 重采样。
   - camber/thickness 低通平滑。
   - 最小厚度夹紧。
   - 尾缘从指定位置平滑融合到目标开口。
6. 调用 XFOIL 跑 alpha sweep。
7. 解析 polar，输出 summary/report。

本次调用：

```bash
/home/vipuser/miniconda3/bin/python scripts/airfoil/run_xfoil_batch.py \
  --xfoil-bin /home/vipuser/tools/xfoil/build/src/xfoil-6.97 \
  --generated-dir outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/geometry_eval_generated_sdf_direct/top_dat \
  --baseline-raw-dir data/airfoil/raw/uiuc \
  --ranked-csv outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/geometry_eval_generated_sdf_direct/rerank_summary.csv \
  --out-dir outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/xfoil_eval_sdf_direct_top32_re1e6_m010_a-2_6 \
  --limit 32 \
  --workers 16 \
  --alpha-start -2 \
  --alpha-end 6 \
  --alpha-step 1 \
  --smooth-passes 8 \
  --te-gap 0.001 \
  --te-blend-start 0.90 \
  --timeout 35 \
  --force
```

工况：

```text
Re = 1e6
Mach = 0.10
Ncrit = 9
alpha = -2 deg 到 6 deg，步长 1 deg
max_iter = 120
```

输出：

```text
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/xfoil_eval_sdf_direct_top32_re1e6_m010_a-2_6/
```

关键产物：

```text
xfoil_summary.csv
xfoil_paired_comparison.csv
xfoil_report.md
xfoil_report.json
cases/{group}__{sample_id}/airfoil.dat
cases/{group}__{sample_id}/polar.csv
cases/{group}__{sample_id}/xfoil.log
```

### 5.2 本次 XFOIL 结果

```text
Generated converged: 32/32
Baseline converged: 31/32

Mean generated best L/D: 107.010
Mean baseline best L/D: 108.799
Mean delta best L/D: -3.061

Mean generated max CL: 0.916
Mean baseline max CL: 0.907
Mean delta max CL: 0.005
```

Top generated by XFOIL best L/D：

| sample_id | best L/D | alpha | CL | CD |
|---|---:|---:|---:|---:|
| ah80129 | 155.314 | 5.00 | 0.9645 | 0.00621 |
| n6h10 | 146.443 | 5.00 | 0.9387 | 0.00641 |
| e583 | 135.387 | 6.00 | 1.0831 | 0.00800 |
| e174 | 134.706 | 5.00 | 0.8931 | 0.00663 |
| ah93157 | 132.890 | 6.00 | 0.9887 | 0.00744 |
| ah79k135 | 132.472 | 5.00 | 0.9379 | 0.00708 |
| fx08s176 | 130.938 | 6.00 | 1.1313 | 0.00864 |
| fx60100sm | 130.495 | 3.00 | 0.7386 | 0.00566 |
| e195 | 128.977 | 5.00 | 0.8951 | 0.00694 |
| ua2-180 | 125.196 | 6.00 | 1.2482 | 0.00997 |

与旧 top50 结果对比，生成翼型平均 best L/D 从约 `30.89` 提升到 `107.01`，说明主要瓶颈确实在 SDF 到翼型坐标的几何重建路径。

### 5.3 XFOIL 结果可视化

调用：

```bash
/home/vipuser/miniconda3/bin/python scripts/airfoil/plot_xfoil_results.py \
  --xfoil-dir outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/xfoil_eval_sdf_direct_top32_re1e6_m010_a-2_6 \
  --top-k 10
```

输出：

```text
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/xfoil_eval_sdf_direct_top32_re1e6_m010_a-2_6/visualizations/
```

包含：

```text
xfoil_convergence_summary.png
xfoil_best_ld_distribution.png
xfoil_paired_metric_scatter.png
xfoil_top_generated_best_ld.png
xfoil_top_generated_polar_curves.png
xfoil_top_generated_airfoil_shapes.png
visualization_index.md
```

这些图分别展示收敛情况、L/D 分布、generated-vs-baseline 散点、top10 L/D 柱状图、top10 极曲线，以及送入 XFOIL 的翼型形状。

## 6. 三维机翼生成模块

核心文件：

```text
scripts/generate_wing.py
```

该脚本用 CadQuery/OpenCascade 从 Selig/UIUC `.dat` 翼型生成一侧梯形机翼 STEP。

### 6.1 功能

输入：

```text
--airfoil     翼型 .dat
--aspect-ratio
--total-length
--dihedral
--root-tip-ratio 或 --taper-ratio
--section-points
--output
--metadata
```

主要步骤：

1. 读取 `.dat`。
2. 归一化翼型。
3. 拆分上下表面。
4. cosine 网格重采样为 CAD-friendly 闭合截面。
5. 根据展弦比、半展长、根梢比和上反角计算平面参数。
6. 构建根部和梢部截面。
7. 用 CadQuery `Solid.makeLoft` 生成 loft solid。
8. 导出 STEP，并写 metadata JSON。

### 6.2 本次调用

使用新采样几何的 top5：

```text
n6h10
goe282
ua2-180
ag455ct02r
oa209
```

单个样例命令：

```bash
/home/vipuser/miniconda3/bin/python scripts/generate_wing.py \
  --airfoil outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/geometry_eval_generated_sdf_direct/top_dat/n6h10.dat \
  --aspect-ratio 8 \
  --total-length 5 \
  --dihedral 5 \
  --output outputs/wing_3d_sdf_direct_top5/n6h10_wing.step \
  --metadata outputs/wing_3d_sdf_direct_top5/n6h10_wing.metadata.json \
  --section-points 161
```

参数：

```text
aspect_ratio = 8
total_length = 5
dihedral = 5 deg
section_points = 161
root_chord = 1.6666666667
tip_chord = 0.8333333333
projected_span = 4.98097
dihedral_height = 0.435779
```

输出目录：

```text
outputs/wing_3d_sdf_direct_top5/
```

包含：

```text
n6h10_wing.step
goe282_wing.step
ua2-180_wing.step
ag455ct02r_wing.step
oa209_wing.step

n6h10_wing.metadata.json
goe282_wing.metadata.json
ua2-180_wing.metadata.json
ag455ct02r_wing.metadata.json
oa209_wing.metadata.json
```

每个 STEP 约 `1.4 MB`，metadata 中 `cadquery_used=true`。

## 7. 模块之间的数据流

完整链路如下：

```text
UIUC/Selig .dat
  -> scripts/prepare_airfoil_uiuc.py
  -> data/airfoil_uiuc_sdf/processed/airfoil_samples/*.npz
  -> AE 训练
  -> PCA latent 构建
  -> PCA-UNet 条件扩散训练
  -> generated_sdf_npz/*.npz
  -> SDF 竖线直接采样
  -> geometry_eval_generated_sdf_direct/top_dat/*.dat
  -> XFOIL 批量校核
  -> xfoil_summary.csv / polar.csv / visualizations
  -> CadQuery 三维 loft
  -> wing_3d_sdf_direct_top5/*.step
```

各模块职责：

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `airfoil_geometry.py` | `.dat` 坐标或数组 | 归一化/重采样翼型、几何条件 | 几何基础工具 |
| `airfoil_sdf.py` | 重采样翼型 | SDF/mask | 训练数据构建 |
| `prepare_airfoil_uiuc.py` | UIUC 原始 `.dat` | `airfoil_samples/*.npz`、索引、条件统计 | 数据集预处理 |
| `train_ae.py` | SDF 数据集 | AE checkpoint、latent records | SDF 压缩 |
| `train_diffusion.py` | AE latent、条件 | UNet checkpoint、生成 SDF、metrics | 条件扩散生成 |
| `evaluate_generated_airfoils.py` | generated SDF | `.dat`、几何误差、rerank | 从 SDF 提取可用翼型 |
| `run_xfoil_batch.py` | `.dat` 与 baseline | polar、summary、report | 气动校核 |
| `plot_xfoil_results.py` | XFOIL 输出 | PNG 可视化 | 结果分析 |
| `generate_wing.py` | `.dat` | STEP、metadata | 3D 机翼建模 |

## 8. 当前结论

1. SDF preview 看起来平滑，但旧 contour 顶点顺序采样会制造翼型坐标突变，这是此前 XFOIL 和 CAD 截面质量差的主因。
2. SDF 竖线直接采样后，247 个生成 SDF 均能导出有效翼型。
3. 新几何路径下 top32 生成翼型 XFOIL 全部收敛，平均 best L/D 达到 `107.01`，接近 baseline 的 `108.80`。
4. 三维重建用同一批新 `.dat` 成功导出 5 个 STEP，说明几何链路已从二维气动校核打通到 3D CAD。
5. 后续最重要的是把“生成 SDF 保存、SDF 直采导出 `.dat`、XFOIL 校核、CAD STEP 输出”整理成稳定的一键流程，减少手工脚本片段。

## 9. 后续建议

1. 将本轮用于 `generated_sdf_npz -> top_dat` 的导出脚本固化为正式 CLI，例如：

   ```bash
   python scripts/airfoil/export_generated_sdf_airfoils.py \
     --sdf-dir .../generated_sdf_npz \
     --out-dir .../geometry_eval_generated_sdf_direct \
     --n-points 257
   ```

2. 在 rerank 中加入光顺指标：

   ```text
   thickness_rms_d2
   thickness_max_d2
   upper_rms_d2
   lower_rms_d2
   camber_rms_d2
   negative_thickness_fraction
   ```

3. XFOIL 结果可以进一步做气动 rerank，例如综合：

   ```text
   condition_score
   best_ld
   max_cl
   convergence_points
   geometry_smoothness_score
   ```

4. `generate_wing.py` 当前仍会在 CAD 路径中重新做一次简单重采样；后续可复用同一套 SDF 直采/光顺几何，确保 XFOIL 与 STEP 截面完全一致。

5. 对 top airfoils 做更多工况的 XFOIL sweep，例如不同 Re、不同 alpha 范围，避免只针对 `Re=1e6, Mach=0.10, alpha=-2..6` 过拟合判断。

6. 若要进一步提升模型本身质量，可考虑让生成端输出参数化翼型表示，例如 CST/PARSEC 或上下表面控制点，SDF 作为辅助监督，而不是唯一几何载体。
