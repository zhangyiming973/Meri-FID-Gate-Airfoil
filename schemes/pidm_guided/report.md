# meri-fid-gate 与 Physics-Informed Diffusion Models 对比分析报告

> 对比对象：[meri-fid-gate](.)（本项目） vs [PhysicsInformedDiffusionModels (PIDM)](https://github.com/jhbastek/PhysicsInformedDiffusionModels)（ICLR 2025，Bastek et al.）

---

## 1. 项目概览

### 1.1 meri-fid-gate（本项目）

**任务**：基于子午面 2D SDF 数据的**尺寸条件生成**——给定 5 维设计参数，生成对应几何形状的符号距离场。

**核心管线**（三阶段）：

| 阶段 | 内容 |
|------|------|
| AE | SDF + 语义 mask → 潜向量 `z_m` (64×16×16) → 重建 SDF |
| 质量门控 | 重建 L1、潜空间 std、通过率校验 |
| 条件扩散 | PCA 压缩潜空间 → MLP/UNet 去噪 → DDIM 采样 → 解码 SDF |

**四种方案**：`mlp` / `pca_unet` / `dim_guided` / `dim_unet`，代码完全独立，统一由 `run.py` 调度。

**条件**：5 维 ConditionVector（轮毂半径、轮缘半径、腹板角、过渡半径、轴向界）。

**已有物理相关能力**：
- AE 阶段：`dimension` 损失（最小壁厚、截面积 L1）
- 扩散阶段：`physics_guidance`（默认 **关闭**），通过 `physics_risk_proxy` 对齐预测 x₀ 与真 x₀ 的壁厚风险
- 配置中有 `sigma_vm_limit`（von Mises 应力上限），**代码尚未接入**

---

### 1.2 PIDM（Physics-Informed Diffusion Models）

**任务**：在**像素/场空间**直接做扩散生成，并通过**控制方程残差**约束生成样本必须满足物理定律。

**典型场景**（论文与代码）：
- **Darcy 流**：64×64 压力场 + 渗透率场，有限差分求 PDE 残差
- **拓扑优化力学**：位移场 + 密度场，FEM（SolidSpy）求平衡方程残差

**核心思想**：将物理约束写成残差 \( r(x_0) \)，在训练损失中加入**虚拟似然项**：

\[
\mathcal{L} = c_{\text{data}} \cdot \mathcal{L}_{\text{DDPM}} \;-\; c_{\text{residual}} \cdot \log p(r=0 \mid x_0^{\text{pred}})
\]

其中方差 \( \text{var}_t \) 与扩散后验方差绑定，随时间步 \( t \) 变化。

**额外机制**：
- **不等式约束** \( c_{\text{ineq}} \)：体积分数、应力上限等
- **优化目标** \( \lambda_{\text{opt}} \)：指数分布似然（如最小化柔度）
- **推理修正**：梯度引导（`residual_grad_guidance`）、CoCoGen 式 \( M/N \) 步校正

---

## 2. 架构对比

```
meri-fid-gate                          PIDM
─────────────────                      ─────────────────
SDF 256×256                            场 64×64（直接扩散）
    ↓ AE 编码                              ↓
z_m 64×16×16                           x_t（像素/场空间）
    ↓ PCA → 128/192 维                       ↓
MLP / UNet 去噪（潜空间）               UNet3D 去噪（场空间）
    ↓ DDIM + CFG                           ↓ DDIM / 多步 x₀ 估计
PCA 逆变换 + AE 解码                   直接输出物理场
    ↓
SDF 256×256
```

| 维度 | meri-fid-gate | PIDM |
|------|---------------|------|
| **扩散空间** | PCA 压缩潜空间（128/192 维） | 原始场空间（64×64×C） |
| **去噪器** | MLP 或 条件 UNet（PCA 网格） | UNet3D（带 Self-Attention） |
| **条件方式** | 5 维设计参数 + CFG | 边界条件 / 载荷 / 体积分数（mechanics） |
| **物理约束形式** | 几何代理量（壁厚、面积） | PDE/FEM 控制方程残差 |
| **物理注入时机** | 训练辅助损失（可选） | 训练主损失 + 推理校正 |
| **物理损失数学形式** | MSE(风险代理) | 高斯虚拟似然 \(-\log p(r=0)\) |
| **数据规模** | 数十～数百样本（single/F404） | 大规模仿真数据集 |
| **两阶段** | AE + 扩散（必须） | 单阶段端到端 |

---

## 3. 相同点

1. **扩散框架一致**：均采用 DDPM 前向加噪 + 噪声/x₀ 预测 + DDIM 加速采样。
2. **Classifier-Free Guidance**：本项目 `cfg_dropout=0.2`、`cfg_scale=1.5`；PIDM 在 mechanics 等场景也有条件 dropout。
3. **在 x₀ 上施加物理约束**：两者都不是只在最终样本上后处理，而是在去噪过程中对 **\( \hat{x}_0 \)** 评估物理量。
4. **时间步相关权重**：PIDM 显式用 `posterior_variance_clipped[t]`；本项目 `physics_guidance` 在后 2/3 epoch 才启用（隐式时间策略）。
5. **工程导向生成**：都面向「生成结果必须满足物理/几何约束」，而非纯视觉质量。

---

## 4. 关键差异

### 4.1 物理约束的「严格程度」

| | meri-fid-gate | PIDM |
|---|---------------|------|
| 约束类型 | 软几何代理（壁厚倒数、面积） | 硬 PDE/FEM 残差 |
| 是否可微贯穿解码 | 是（AE 冻结但可反传） | 直接在输出场上计算 |
| 应力/平衡方程 | 配置预留，未实现 | mechanics 完整 FEM 残差 |
| 训练目标 | 对齐「真 x₀ 的风险」 | 压残差趋近 0 |

本项目当前的 `physics_guidance` 本质是 **「让预测 x₀ 的物理代理与真实 x₀ 一致」**，而非 **「让预测 x₀ 满足独立物理定律」**。当真实样本本身不满足约束时，这会传播数据偏差。

### 4.2 潜空间 vs 像素空间

PIDM 在 **64×64 场** 上直接扩散，物理残差与网络输出同空间。

本项目在 **128/192 维 PCA 空间** 扩散，物理量必须在 **解码后的 SDF** 上计算：

```python
# 现有实现（schemes/mlp/train/train_diffusion.py）
x0_hat_enc = schedule.predict_x0(xt, t, pred)
z_raw_hat = codec.decode_to_raw(x0_hat_enc)
risk_pred = physics_risk_proxy(ae.decode(z_raw_hat))  # 需反传穿过 AE
```

这带来：
- **梯度路径更长**（PCA 逆变换 + Conv AE 解码）
- **物理信号可能被 AE 平滑/失真**
- 但 **计算量远小于** 256×256 像素扩散

### 4.3 条件语义不同

- **meri-fid-gate**：5 维连续设计参数（尺寸引导），`dim_guided`/`dim_unet` 还有 ConditionVector 适用性校验。
- **PIDM**：边界条件、载荷图像、体积分数等**场级条件**，mechanics 任务中条件与物理求解强耦合。

### 4.4 推理阶段

| | meri-fid-gate | PIDM |
|---|---------------|------|
| 采样 | DDIM 50/80 步 + CFG | DDIM + 可选多步 x₀ 估计 |
| 推理物理修正 | **无** | 梯度引导、CoCoGen M/N 校正 |
| 残差监控 | 测试 L1 指标 | 采样残差均值/中位数 |

---

## 5. 本项目现有物理能力盘点

### 5.1 AE 阶段（已生效）

`schemes/mlp/losses/ae_losses.py` 中：

- `estimate_min_thickness` / `estimate_area`：从 SDF 启发式估计几何量
- `dimension` 损失：与 NPZ 中 `min_thickness_mm`、`area_mm2` 对齐
- `physics_risk_proxy`：壁厚越小风险越高（供扩散阶段使用）

### 5.2 扩散阶段（已实现但未默认开启）

`schemes/mlp/train/train_diffusion.py` 中：

```python
if pg_on and epoch > diff_cfg["epochs"] // 3:
    x0_hat_enc = schedule.predict_x0(xt, t, pred)
    z_raw_hat = codec.decode_to_raw(x0_hat_enc)
    risk_pred = physics_risk_proxy(ae.decode(z_raw_hat))
    with torch.no_grad():
        z0_raw = codec.decode_to_raw(z0)
        risk_gt = physics_risk_proxy(ae.decode(z0_raw))
    loss = loss + pg_w * torch.nn.functional.mse_loss(risk_pred, risk_gt)
```

**局限**：
1. 对齐的是「风险代理」，不是独立物理残差
2. `risk_gt` 来自真实样本，不保证物理最优
3. `sigma_vm_limit` 未使用
4. 仅在训练期生效，采样时无校正

---

## 6. 能否直接使用 PIDM 方案？

### 6.1 结论：**不能整库照搬，方法论可借鉴**

| 可直接复用 | 不能直接复用 |
|-----------|-------------|
| 虚拟似然损失形式 | Darcy/FEM 残差模块 |
| x₀ 估计 + 残差评估流程 | UNet3D 架构与 64×64 数据管线 |
| 推理时梯度引导 / CoCoGen 思路 | mechanics 的 SolidSpy/FEM 依赖 |
| `c_residual` / `c_ineq` / `lambda_opt` 配置模式 | LMDB 数据集与 train_pairs 逻辑 |

**根本原因**：领域不同（子午面 SDF 几何 vs 流场/拓扑优化场），数据结构不同（NPZ + 潜空间 vs CSV/FEM 场），控制方程不同（几何约束 vs PDE/FEM）。

### 6.2 推荐集成策略（由易到难）

#### 方案 A：激活并改进现有 `physics_guidance`（低成本，1–2 天）

将 MSE 风险对齐改为 **独立几何残差**：

```python
def geometry_residual(sdf, t_min_mm=2.0, area_target=None, sdf_scale=12.0):
    thick = estimate_min_thickness(sdf, sdf_scale)
    r_thick = F.relu(t_min_mm - thick)                    # 壁厚不足
    r_area = torch.tensor(0.0, device=sdf.device)
    if area_target is not None:
        r_area = (estimate_area(sdf, sdf_scale) - area_target).abs()
    return torch.stack([r_thick, r_area], dim=-1)         # (B, 2)
```

配置扩展：

```json
"physics_guidance": {
  "enabled": true,
  "mode": "residual_mse",
  "weight": 0.02,
  "min_thickness_mm": 2.0,
  "use_area_constraint": false
}
```

#### 方案 B：PIDM 式虚拟似然（推荐，3–5 天）

参考 PIDM 核心公式，在 `train_diffusion.py` 中替换辅助损失：

```python
# PIDM: loss += -c_residual * gaussian_log_likelihood(0, mean=residual, var=posterior_var[t])
residual = geometry_residual(ae.decode(z_raw_hat), t_min=pg_cfg["min_thickness_mm"])
var = schedule.posterior_variance[t].view(-1, 1)  # 需在 DiffusionSchedule 中暴露
physics_loss = -pg_cfg["weight"] * gaussian_log_likelihood(
    torch.zeros_like(residual), means=residual, variance=var
).mean()
loss = loss + physics_loss
```

**与 PIDM 对齐的关键点**：
- 残差目标为 **0**（而非对齐 GT 风险）
- 方差随 \( t \) 缩放（高噪声步容忍更大残差）
- 可选 `x0_estimation: mean | sample`（高噪声步用 DDIM 子步估计 x₀，见 PIDM `model.yaml`）

#### 方案 C：SDF 微分物理约束（中期，1–2 周）

子午面 SDF 可加入 **Eikonal 型约束**（|\nabla SDF| ≈ 1），用有限差分在解码 SDF 上求残差：

```python
def eikonal_residual(sdf):  # sdf: (B,1,H,W)
    dx = sdf[:, :, 1:, :] - sdf[:, :, :-1, :]
    dy = sdf[:, :, :, 1:] - sdf[:, :, :, :-1]
    grad_norm = torch.sqrt(dx[:, :, :-1, :]**2 + dy[:, :, :, :-1]**2 + 1e-6)
    return (grad_norm - 1.0).abs().mean(dim=(-2, -1))
```

这与 PIDM 在 Darcy 上用 `GradientsHelper` 求 PDE 导数的做法同构，但方程换成 SDF 合法性的 Eikonal 条件。

#### 方案 D：推理时物理校正（中长期）

参考 PIDM 的 `M_correction` / `N_correction` 与 `residual_grad_guidance`，在 `ddim_sample_cfg` 每步或末步：

```python
# 伪代码：对 x0_hat 做一步梯度下降减小几何残差
x0_hat = schedule.predict_x0(xt, t, eps_pred)
sdf_hat = ae.decode(codec.decode_to_raw(x0_hat))
res = geometry_residual(sdf_hat).sum()
grad = torch.autograd.grad(res, x0_hat)[0]
x0_hat_corrected = x0_hat - alpha * grad
# 继续 DDIM 步
```

注意：潜空间维度低（128/192），校正比 4096 维像素空间更稳定，但需控制步长避免破坏扩散流形。

#### 方案 E：完整应力约束（长期，需外部求解器）

若需使用配置中的 `sigma_vm_limit`，需：
1. 2D SDF → 3D 回转体网格
2. 调用 FEM（类似 PIDM 的 SolidSpy）求 von Mises 应力
3. 残差 `relu(sigma_vm - sigma_limit)`

计算成本高，建议仅作 **推理后验校验** 或 **少量样本精修**，不宜每训练 step 调用。

---

## 7. 推荐实施路线

```
Phase 1（立即可做）
├── 开启 physics_guidance，对比 single 数据集 gen L1
├── 将 MSE(risk_pred, risk_gt) 改为 geometry_residual → 0
└── 在 collect-timing / test 报告中增加物理残差指标

Phase 2（PIDM 核心迁移）
├── 实现 gaussian_log_likelihood + posterior_variance 绑定
├── 抽象 PhysicsResidual 接口（类似 PIDM ResidualsDarcy）
├── 支持 c_residual / c_ineq 配置（壁厚、面积、Eikonal）
└── 在 dim_unet 方案上先验证（UNet + 192 维，与 PIDM 架构最接近）

Phase 3（推理增强）
├── DDIM 采样中加入 N 步 x₀ 残差校正
└── 可选：CFG 与物理梯度引导联合（参考 PIDM residual_grad_guidance）

Phase 4（可选）
└── FEM 应力后验验证（sigma_vm_limit）
```

---

## 8. 方案选型建议

| 你的目标 | 建议 |
|---------|------|
| 快速验证物理引导是否有用 | **方案 A**：改残差 + 开启 `physics_guidance` |
| 与 PIDM 论文方法对齐、可发表 | **方案 B + C**：虚拟似然 + SDF Eikonal |
| 生成结果壁厚/面积硬约束 | **方案 B**（训练）+ **方案 D**（推理校正） |
| von Mises 应力约束 | **方案 E**，需独立 FEM 模块，不能直接用 PIDM 代码 |
| 复用 PIDM 仓库 | **不建议**；仅参考 `src/denoising_utils.py` 中 `model_estimation_loss` 与 `model.yaml` 参数设计 |

---

## 9. 总结

| 问题 | 答案 |
|------|------|
| 两项目本质相似吗？ | **框架相似**（DDPM/DDIM + 条件生成 + 物理约束），**问题域与约束形式差异大** |
| 本项目缺什么？ | 独立物理残差、虚拟似然形式、推理时校正、应力/FEM 求解 |
| 本项目有什么优势？ | 潜空间扩散高效、尺寸条件明确、AE 门控保证潜空间质量、四套方案可对比 |
| 能否直接用 PIDM？ | **不能整库使用**；**损失设计、残差接口、推理校正思路可直接借鉴** |
| 最小改动路径？ | 改 `physics_guidance` 为「残差→0」+ 开启配置，在 `mlp` 或 `dim_unet` 上 A/B 测试 |

---

## 10. 参考文献

- Bastek et al., *Physics-Informed Diffusion Models*, ICLR 2025. [GitHub](https://github.com/jhbastek/PhysicsInformedDiffusionModels) | [OpenReview](https://openreview.net/forum?id=tpYeermigp)
- 本项目扩散实现：`schemes/*/models/diffusion.py`、`schemes/*/train/train_diffusion.py`
- PIDM 核心训练损失：`src/denoising_utils.py::model_estimation_loss`
- PIDM 配置范例：`model.yaml`（`c_residual`, `c_ineq`, `lambda_opt`, `correction_mode`）
