"""MLP 扩散方案数据模块：条件列定义与数据集加载。

条件向量描述子午面几何设计参数（轮毂、轮缘、腹板等），
供自编码器与 MLP 条件扩散模型使用。
"""

# 默认 5 维设计条件列（与 data/*/dataset.json 中 condition_columns 一致）
CONDITION_COLUMNS = [
    "hub_r_end_mm",
    "rim_r_start_mm",
    "angle_web_deg",
    "r_trans_bore_web_mm",
    "z_min",
]
