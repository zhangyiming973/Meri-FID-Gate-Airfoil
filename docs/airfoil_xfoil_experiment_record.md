# UIUC 生成翼型 XFOIL 气动校核实验记录

记录日期：2026-06-15  
分支：`airfoil-conditional-diffusion`  
目标：拉取并构建 XFOIL，对条件扩散生成翼型进行直接气动校核，并与 UIUC 原始同名翼型对比。

## 1. 背景

前序实验已完成：

- Stage A 数据预处理：`1646/1665` 通过，`1399 train / 247 test`
- Stage B AE 训练：`val L1=0.00354`，几何 pass ratio `100%`
- Stage C 扩散训练：`mean gen L1=0.0374`
- 几何重排输出目录：

```text
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/geometry_eval_rerank_c32/
```

几何重排后生成翼型整体合法，但用户希望进一步判断它们是否真正适合气动校核，并要求与原 UIUC 数据集做对比。

## 2. XFOIL 环境准备

本机初始状态：

- `xfoil` 不存在
- `gfortran` 不存在
- `cmake` 不存在
- `make` 存在
- 系统 `sudo apt-get` 不可用：`sudo` 需要密码/TTY

处理方式：

1. 用用户可写 Conda 环境安装构建工具，避免修改不可写的 `jaxfem` 环境。

```bash
/home/vipuser/miniconda3/bin/conda create -y \
  -p /home/vipuser/conda_envs/xfoil-build \
  -c conda-forge gfortran_linux-64 cmake make
```

2. XFOIL 构建缺少 X11 头文件，补充依赖：

```bash
/home/vipuser/miniconda3/bin/conda install -y \
  -p /home/vipuser/conda_envs/xfoil-build \
  -c conda-forge xorg-libx11 xorg-xorgproto
```

3. 拉取源码：

```bash
git clone https://github.com/RobotLocomotion/xfoil /home/vipuser/tools/xfoil
```

说明：严格意义上 Mark Drela 的 XFOIL 官方源头是 MIT 页面 `web.mit.edu/drela/Public/web/xfoil/`，不是 GitHub 仓库。本次使用的是 `RobotLocomotion/xfoil` fork，README 指向 MIT Drela XFOIL 页面，便于 CMake 构建。

源码 commit：

```text
d11a1544b53623c01bb3b0cceb5862311be0e1f8
```

4. CMake 配置和编译：

```bash
/home/vipuser/conda_envs/xfoil-build/bin/cmake \
  -S /home/vipuser/tools/xfoil \
  -B /home/vipuser/tools/xfoil/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/home/vipuser/tools/xfoil/install \
  -DCMAKE_Fortran_COMPILER=/home/vipuser/conda_envs/xfoil-build/bin/x86_64-conda-linux-gnu-gfortran \
  -DCMAKE_C_COMPILER=/home/vipuser/conda_envs/xfoil-build/bin/x86_64-conda-linux-gnu-gcc \
  -DCMAKE_PREFIX_PATH=/home/vipuser/conda_envs/xfoil-build \
  -DCMAKE_INCLUDE_PATH=/home/vipuser/conda_envs/xfoil-build/include \
  -DCMAKE_LIBRARY_PATH=/home/vipuser/conda_envs/xfoil-build/lib

/home/vipuser/conda_envs/xfoil-build/bin/cmake \
  --build /home/vipuser/tools/xfoil/build -j 4
```

生成可执行文件：

```text
/home/vipuser/tools/xfoil/build/src/xfoil-6.97
```

## 3. 批量校核脚本

新增脚本：

```text
scripts/airfoil/run_xfoil_batch.py
```

提交：

```text
9a6f5f8 Add batch XFOIL airfoil evaluation
```

脚本能力：

- 读取几何重排 `rerank_summary.csv`
- 按排名收集生成 `.dat`
- 在 UIUC 原始目录中查找同名 baseline `.dat`
- 对生成翼型和原始同名翼型运行相同 XFOIL 工况
- 支持 CPU 多进程并行，使用 `--workers`
- 自动关闭 XFOIL 图形输出，适配无 DISPLAY 环境
- 解析 polar 文件并输出：
  - `xfoil_summary.csv`
  - `xfoil_paired_comparison.csv`
  - `xfoil_report.json`
  - `xfoil_report.md`

## 4. 关键工程问题与修正

### 4.1 XFOIL 长路径读取失败

最初 XFOIL 对长路径 `LOAD <long/path/file.dat>` 解析异常，日志里把文件名读成单字符。

修正：

- 每个 case 创建独立工作目录
- 把输入 `.dat` 复制/转换成短文件名：

```text
airfoil.dat
```

### 4.2 无显示环境打开 X11 失败

XFOIL 默认启用 X11 图形，日志出现：

```text
Cannot open display...aborting
```

修正：

在批处理输入开头加入：

```text
PLOP
G

```

将 graphics flag 从 `True` 切换为 `False`。

### 4.3 生成翼型点数超过 XFOIL 内部限制

生成 `.dat` 初始为 `513` 个点，XFOIL 日志显示当前 airfoil node 上限不足，并出现异常几何解释，例如弦长被识别为 `0.010`、厚度异常。

修正：

- 对送入 XFOIL 的临时几何统一做 cosine 重采样
- 默认 `161` 个 x 点/面，输出 `321` 个坐标点
- 低于 XFOIL 当前构建的节点限制

### 4.4 生成翼型 viscous BL 不收敛

即使重采样后，部分生成翼型仍在 XFOIL 粘性边界层求解中出现 `NaN` 或不收敛。原因主要是 SDF 轮廓提取带来的局部锯齿、尖角和尾缘区域不光顺。

加入仅用于 XFOIL 输入的几何清洗：

- camber/thickness 线平滑
- 最小厚度夹紧
- 尾缘开口渐变闭合，而不是只改最后一个点
- 尾缘从 `x/c=0.90` 开始平滑过渡到 `0.001c`

注意：这些处理只写入 XFOIL case 临时目录，不回写生成翼型原始 `.dat`。

## 5. 最终校核设置

最终 top50 校核命令：

```bash
/home/vipuser/miniconda3/bin/python scripts/airfoil/run_xfoil_batch.py \
  --xfoil-bin /home/vipuser/tools/xfoil/build/src/xfoil-6.97 \
  --generated-dir outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/geometry_eval_rerank_c32/top_dat \
  --baseline-raw-dir data/airfoil/raw/uiuc \
  --ranked-csv outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/geometry_eval_rerank_c32/rerank_summary.csv \
  --out-dir outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/xfoil_eval_top50_re1e6_m010_a-2_6 \
  --limit 50 \
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

- Reynolds number: `1e6`
- Mach: `0.10`
- Ncrit: `9`
- alpha sweep: `-2` 到 `6` 度，步长 `1` 度
- max iter: `120`
- 并行：`16` workers

结果目录：

```text
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/xfoil_eval_top50_re1e6_m010_a-2_6/
```

## 6. Top50 结果

汇总：

| 指标 | 生成翼型 | UIUC 原始同名翼型 |
|---|---:|---:|
| 有效 polar | 40/50 | 41/50 |
| 完整 alpha sweep | 9/50 | 23/50 |
| 平均 best L/D | 30.89 | 102.71 |
| 中位 best L/D | 23.54 | 96.34 |
| 平均 max CL | 0.553 | 0.921 |

有效配对样本：

| 指标 | 值 |
|---|---:|
| 有效配对数 | 32 |
| 平均 delta best L/D | -73.42 |
| 生成优于原始的配对数 | 1/32 |

结论：生成翼型经过几何清洗后可以进入 XFOIL，但整体气动性能明显低于 UIUC 原始同名翼型。

## 7. 表现最好的生成翼型

按生成翼型 best L/D 排序：

| rank | sample_id | best L/D | alpha at best L/D | max CL | 有效点数 |
|---:|---|---:|---:|---:|---:|
| 1 | goe188 | 103.41 | 5.0 | 0.930 | 9 |
| 2 | m14 | 94.14 | 6.0 | 0.915 | 9 |
| 3 | fx74cl6140 | 88.47 | 6.0 | 1.082 | 9 |
| 4 | goe346 | 76.16 | 1.0 | 0.224 | 4 |
| 5 | goe322 | 74.69 | 5.0 | 1.065 | 8 |
| 6 | ua2-180 | 57.16 | 3.0 | 0.879 | 9 |
| 7 | pmc19sm | 55.77 | 6.0 | 0.863 | 8 |
| 8 | rc08n1 | 49.66 | 6.0 | 0.781 | 9 |
| 9 | fx66h60 | 48.82 | 6.0 | 0.707 | 8 |
| 10 | isa960 | 47.86 | 2.0 | 0.613 | 6 |
| 11 | n6h10 | 47.10 | 2.0 | 0.584 | 7 |
| 12 | goe368 | 46.52 | 3.0 | 0.577 | 7 |
| 13 | goe701 | 40.44 | 4.0 | 0.772 | 8 |
| 14 | n11h9 | 33.36 | 6.0 | 0.855 | 8 |
| 15 | goe746 | 32.59 | 3.0 | 0.789 | 9 |

建议优先进入后续 CFD 或更宽攻角复核的生成样本：

```text
goe188
m14
fx74cl6140
goe346
goe322
ua2-180
pmc19sm
rc08n1
```

## 8. 解释与判断

这轮结果说明：

1. 当前生成翼型的几何合法性不等于气动可用性。
2. 生成翼型在 SDF/轮廓空间上接近目标条件，但局部曲率、尾缘区域、厚度和弯度分布仍会显著影响 XFOIL 粘性求解。
3. 几何清洗可以把很多生成翼型送入 XFOIL，但不能弥补整体气动性能差距。
4. 原始 UIUC 同名翼型也并非全部 XFOIL 收敛，说明 XFOIL 对低质量、特殊或粗糙点列本身敏感；但原始数据的平均 L/D 和 max CL 仍明显更高。

## 9. 后续最高收益方向

下一轮优化建议按收益排序：

1. 加入 XFOIL/曲率友好的几何后处理
   - 表面曲率平滑
   - 尾缘厚度约束
   - 前缘半径和尾缘楔角约束
   - 输出点列重采样为 XFOIL/CFD 友好的格式

2. 训练目标从 SDF 像素误差扩展到表面几何
   - 上下表面 y 坐标损失
   - thickness/camber 分布损失
   - 曲率或二阶差分正则
   - 尾缘开口和尾缘角损失

3. 采样后加入气动 rerank/rejection
   - 先几何过滤
   - 再 XFOIL 快速筛选
   - 以 XFOIL 成功率、best L/D、max CL 和目标条件误差综合排序

4. 对 top 样本做更高保真 CFD
   - XFOIL 作为第一道筛选
   - CFD 对 `goe188`, `m14`, `fx74cl6140`, `goe322` 等进一步验证

## 10. 产物清单

代码：

```text
scripts/airfoil/run_xfoil_batch.py
```

XFOIL：

```text
/home/vipuser/tools/xfoil
/home/vipuser/tools/xfoil/build/src/xfoil-6.97
/home/vipuser/conda_envs/xfoil-build
```

结果：

```text
outputs/pca_unet/airfoil_uiuc_sdf/20260615_114115/xfoil_eval_top50_re1e6_m010_a-2_6/
  xfoil_report.md
  xfoil_report.json
  xfoil_summary.csv
  xfoil_paired_comparison.csv
  xfoil_conclusion_cn.md
  cases/
```

Git 提交：

```text
9a6f5f8 Add batch XFOIL airfoil evaluation
```

本记录文件：

```text
docs/airfoil_xfoil_experiment_record.md
```
