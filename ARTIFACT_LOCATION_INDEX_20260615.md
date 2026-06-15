# Meri-FID-Gate-Airfoil 产物位置索引

日期：2026-06-15  
项目根目录：`/home/vipuser/Meri-FID-Gate-Airfoil`

本文档用于快速定位本轮实验的关键实验记录、中间数据、可视化图像和 STEP 文件。

## 1. 总结与实验记录

项目级完整总结：

```text
AIRFOIL_PIPELINE_EXPERIMENT_SUMMARY_20260615.md
```

本轮 SDF 采样修正实验日志：

```text
docs/airfoil_sdf_sampling_experiment_20260615.md
```

此前 XFOIL 实验记录：

```text
docs/airfoil_xfoil_experiment_record.md
```

完整项目总结与实验报告：

```text
项目总结与实验报告_20260615.md
```

运行手册与准备步骤：

```text
docs/airfoil_gpu_runbook.md
docs/airfoil_experiment_steps.md
```

## 2. 核心代码模块

几何解析、归一化、上下表面重采样、条件计算：

```text
scripts/airfoil_geometry.py
```

翼型 SDF 栅格化：

```text
scripts/airfoil_sdf.py
```

UIUC/Selig 数据预处理：

```text
scripts/prepare_airfoil_uiuc.py
```

生成 SDF 到翼型 `.dat` 的几何评估与导出，包含本轮新增的 SDF 竖线直接采样逻辑：

```text
scripts/airfoil/evaluate_generated_airfoils.py
```

XFOIL 批量校核：

```text
scripts/airfoil/run_xfoil_batch.py
```

XFOIL 结果可视化：

```text
scripts/airfoil/plot_xfoil_results.py
```

CadQuery/OpenCascade 三维机翼 STEP 生成：

```text
scripts/generate_wing.py
```

PCA-UNet 训练与生成：

```text
schemes/pca_unet/train/train_ae.py
schemes/pca_unet/train/train_diffusion.py
schemes/pca_unet/models/
schemes/pca_unet/train/diffusion_codec.py
```

## 3. 原始与预处理数据

UIUC 原始翼型库：

```text
data/airfoil/raw/uiuc/
```

翼型 SDF 数据集配置：

```text
data/airfoil_uiuc_sdf/dataset.json
```

预处理后的翼型 SDF 样本：

```text
data/airfoil_uiuc_sdf/processed/airfoil_samples/*.npz
```

预处理索引、筛选报告、条件统计：

```text
data/airfoil_uiuc_sdf/processed/airfoil_index.csv
data/airfoil_uiuc_sdf/processed/filter_report.json
data/airfoil_uiuc_sdf/processed/condition_stats.json
data/airfoil_uiuc_sdf/processed/train_split.csv
data/airfoil_uiuc_sdf/processed/test_split.csv
```

## 4. 模型训练与扩散生成输出

AE 训练输出：

```text
outputs/pca_unet/airfoil_uiuc_sdf/20260615_110639/autoencoder/
```

扩散训练与测试生成输出：

```text
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/diffusion/
```

关键文件：

```text
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/diffusion/best_unet_diffusion.pt
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/diffusion/latent_pca.json
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/diffusion/summary.json
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/diffusion/generation_metrics.json
```

生成样本 SDF 数值数据，本轮后续几何导出使用此目录：

```text
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/diffusion/generated_sdf_npz/*.npz
```

每个 `.npz` 包含：

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

扩散生成 SDF 面板图：

```text
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/diffusion/generations/*.png
```

## 5. 新 SDF 直采几何导出结果

本轮最重要的几何导出目录：

```text
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/geometry_eval_generated_sdf_direct/
```

关键文件：

```text
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/geometry_eval_generated_sdf_direct/geometry_eval_report.json
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/geometry_eval_generated_sdf_direct/rerank_summary.csv
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/geometry_eval_generated_sdf_direct/per_sample_geometry.csv
```

新采样导出的生成翼型 `.dat`：

```text
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/geometry_eval_generated_sdf_direct/top_dat/*.dat
```

候选翼型 `.dat`：

```text
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/geometry_eval_generated_sdf_direct/candidate_dat/{sample_id}/rank01_cand000.dat
```

本次导出统计：

```text
num_inputs   = 247
num_airfoils = 247
num_valid    = 247
num_failures = 0
```

## 6. SDF 采样逻辑预览图

生成样本 SDF 的新旧采样对比图：

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

数据集已保存 SDF 的新旧采样对比图：

```text
outputs/airfoil_sdf_sampling_preview_20260615/
```

包含：

```text
goe188_saved_sdf_sampling.png
m14_saved_sdf_sampling.png
fx74cl6140_saved_sdf_sampling.png
ua2-180_saved_sdf_sampling.png
rc08n1_saved_sdf_sampling.png
```

旧 raw generated 与 XFOIL 临时清洗后的对比图：

```text
outputs/airfoil_quality_preview_20260615/
```

包含：

```text
goe188_raw_vs_clean.png
m14_raw_vs_clean.png
fx74cl6140_raw_vs_clean.png
ua2-180_raw_vs_clean.png
rc08n1_raw_vs_clean.png
```

## 7. XFOIL 校核结果

本轮最终 XFOIL top32 校核目录：

```text
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/xfoil_eval_sdf_direct_top32_re1e6_m010_a-2_6/
```

关键报告：

```text
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/xfoil_eval_sdf_direct_top32_re1e6_m010_a-2_6/xfoil_report.md
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/xfoil_eval_sdf_direct_top32_re1e6_m010_a-2_6/xfoil_report.json
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/xfoil_eval_sdf_direct_top32_re1e6_m010_a-2_6/xfoil_summary.csv
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/xfoil_eval_sdf_direct_top32_re1e6_m010_a-2_6/xfoil_paired_comparison.csv
```

每个样本的 XFOIL 输入、日志和 polar：

```text
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/xfoil_eval_sdf_direct_top32_re1e6_m010_a-2_6/cases/{group}__{sample_id}/airfoil.dat
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/xfoil_eval_sdf_direct_top32_re1e6_m010_a-2_6/cases/{group}__{sample_id}/polar.csv
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/xfoil_eval_sdf_direct_top32_re1e6_m010_a-2_6/cases/{group}__{sample_id}/xfoil.log
```

本轮 XFOIL 结果摘要：

```text
Generated converged: 32/32
Baseline converged: 31/32
Mean generated best L/D: 107.010
Mean baseline best L/D: 108.799
Mean delta best L/D: -3.061
Mean generated max CL: 0.916
Mean baseline max CL: 0.907
```

## 8. XFOIL 可视化图像

XFOIL 可视化目录：

```text
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/xfoil_eval_sdf_direct_top32_re1e6_m010_a-2_6/visualizations/
```

索引：

```text
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/xfoil_eval_sdf_direct_top32_re1e6_m010_a-2_6/visualizations/visualization_index.md
```

图像：

```text
xfoil_convergence_summary.png
xfoil_best_ld_distribution.png
xfoil_paired_metric_scatter.png
xfoil_top_generated_best_ld.png
xfoil_top_generated_polar_curves.png
xfoil_top_generated_airfoil_shapes.png
```

## 9. 3D 机翼 STEP 文件

本轮用新 SDF 直采 `.dat` 生成的 top5 三维机翼目录：

```text
outputs/wing_3d_sdf_direct_top5/
```

STEP 文件：

```text
outputs/wing_3d_sdf_direct_top5/n6h10_wing.step
outputs/wing_3d_sdf_direct_top5/goe282_wing.step
outputs/wing_3d_sdf_direct_top5/ua2-180_wing.step
outputs/wing_3d_sdf_direct_top5/ag455ct02r_wing.step
outputs/wing_3d_sdf_direct_top5/oa209_wing.step
```

对应 metadata：

```text
outputs/wing_3d_sdf_direct_top5/n6h10_wing.metadata.json
outputs/wing_3d_sdf_direct_top5/goe282_wing.metadata.json
outputs/wing_3d_sdf_direct_top5/ua2-180_wing.metadata.json
outputs/wing_3d_sdf_direct_top5/ag455ct02r_wing.metadata.json
outputs/wing_3d_sdf_direct_top5/oa209_wing.metadata.json
```

生成参数：

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

## 10. 旧实验结果位置

旧几何重排和 XFOIL 结果仍保留，可用于对比：

```text
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/geometry_eval_rerank_c1/
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/geometry_eval_rerank_c32/
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/xfoil_eval_top50_re1e6_m010_a-2_6/
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/xfoil_eval_top20_re1e6_m010_a-2_6/
```

旧 top50 生成翼型平均 best L/D 约 `30.89`，本轮新 SDF 直采 top32 平均 best L/D 为 `107.01`。

## 11. 打包说明

本项目完整目录已打包为：

```text
/home/vipuser/Meri-FID-Gate-Airfoil_20260615_full.zip
```

压缩包包含代码、数据、模型输出、中间数据、可视化图像、XFOIL 结果和 STEP 文件。
