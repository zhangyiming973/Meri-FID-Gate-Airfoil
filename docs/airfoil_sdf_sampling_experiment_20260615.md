# Airfoil SDF Sampling Experiment - 2026-06-15

## 背景

本次检查针对生成翼型导出 `.dat` 后出现的几何质量问题：

1. raw generated 翼型存在局部突变，后续 XFOIL/CAD 清洗只能缓解，清洗后的厚度线仍可能保留突变。
2. 即使没有明显跳变，raw 轮廓也存在很多尖峰和坑洼，导致送入 XFOIL 的气动表现变差，也会让 CadQuery 生成的 STEP 截面不光顺。

对比 SDF preview 后判断：SDF 场本身没有同等程度的突跃，问题主要来自 `SDF -> contour vertices -> split_surfaces -> np.interp` 的采样链路。

## 原因判断

旧逻辑位于 `scripts/airfoil/evaluate_generated_airfoils.py`：

- `_extract_largest_contour` 使用 `matplotlib.contour` 提取零等值线顶点。
- `_contour_to_airfoil` 将 contour 归一化后交给 `split_surfaces`。
- `split_surfaces` 按前缘索引把一个闭合点列切成上下两段。

该方法隐含假设 contour 顶点顺序稳定、连续、无局部跳段。生成 SDF 的零等值线如果存在局部毛刺、多段近接、顶点顺序异常，就会在切分和线性插值时放大为翼型表面的突变。

## 修改方案

已在 `scripts/airfoil/evaluate_generated_airfoils.py` 中新增 SDF 直采路径：

- `_interp_sdf_column`：在物理 `x` 坐标处插值取得一列 SDF。
- `_zero_crossings`：沿 `y` 方向查找零交点。
- `_surface_pair_from_sdf_column`：选择同一竖线上的主翼型上下表面。
- `_sample_sdf_airfoil`：只用 contour 估计弦向范围，随后按 cosine `x/c` 网格逐列从 SDF 直接采样上下表面。
- `_despike_line`：对 camber/thickness 做轻量二阶差分尖峰替换。

`_candidate_from_sdf` 现在优先使用 `_sample_sdf_airfoil`，失败时回退到旧的 `_contour_to_airfoil`，避免单个坏样本中断批处理。

同时新增 CLI 参数：

```bash
--limit-samples N
```

用于只评估前 N 个测试样本，便于做小规模预览。

## 测试

新增测试：

```text
tests/test_airfoil_cleanup_rerank.py::test_sample_sdf_airfoil_ignores_contour_order_spike
```

该测试构造一个光顺 SDF，同时人为给 contour 加入顺序跳变点，验证新采样逻辑不跟随坏 contour 产生表面突变。

已运行：

```bash
/home/vipuser/miniconda3/bin/python -m pytest tests/test_airfoil_cleanup_rerank.py
```

结果：

```text
4 passed
```

## 预览产物

### 1. XFOIL 清洗前后对比

目录：

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

该预览显示：XFOIL 清洗能显著降低粗糙度，但如果 raw generated 本身有突变，清洗后仍可能留下厚度突变。

### 2. 已保存 SDF 的采样逻辑对比

目录：

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

图中含义：

- 灰色：SDF 零等值线
- 红色：旧 contour 顶点顺序采样
- 蓝色：新 SDF 竖线直接采样

二阶差分粗糙度对比：

| sample_id | old thickness RMS d2 | new thickness RMS d2 | old max d2 | new max d2 |
|---|---:|---:|---:|---:|
| goe188 | 0.00684473 | 0.000231859 | 0.0483055 | 0.00363288 |
| m14 | 0.0123598 | 0.000211013 | 0.063159 | 0.00328711 |
| fx74cl6140 | 0.0327578 | 0.000821645 | 0.107824 | 0.0130698 |
| ua2-180 | 0.0410126 | 0.000817907 | 0.15728 | 0.0130069 |
| rc08n1 | 0.00953971 | 0.000388069 | 0.0546611 | 0.00617593 |

结论：SDF 直采明显减少由 contour 顺序和线性插值造成的跳变/尖峰。

## 当前限制

此前扩散生成阶段保存了：

```text
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/diffusion/generations/*.png
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/diffusion/generation_metrics.json
```

但没有保存数值版 `sdf_gen` 数组。训练代码只在 `schemes/pca_unet/train/train_diffusion.py` 中将 `sdf_gen` 画成 PNG，并把 L1/条件写入 JSON。因此本次第二组预览使用的是数据集已保存的 ground-truth SDF：

```text
data/airfoil_uiuc_sdf/processed/airfoil_samples/*.npz
```

它验证的是采样逻辑，不是完整扩散生成样本的重新导出。

## 后续建议

1. 在扩散测试/生成阶段增加可选保存数值 SDF：

   ```text
   diffusion/generated_sdf_npz/{sample_id}.npz
   ```

   建议保存 `sdf_gen`、`sdf_gt`、condition、sample_id。

2. 用已保存的 `sdf_gen` 数组重新跑 `_sample_sdf_airfoil`，导出新的 `top_dat`。

3. 将 CADQuery 和 XFOIL 都改为读取同一套新导出的 `.dat`，避免 XFOIL 临时清洗与 STEP 截面来源不一致。

4. 在 rerank 中加入几何光顺指标，例如：

   - `thickness_rms_d2`
   - `thickness_max_d2`
   - `upper/lower max d2`
   - 局部负厚度比例

5. 完成后重新跑小批量 XFOIL 与 STEP 预览，再决定是否全量重排。
