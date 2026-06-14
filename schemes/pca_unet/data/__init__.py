"""PCA-UNet 数据模块：条件列定义与数据集加载。

条件向量描述子午面几何设计参数，用于条件扩散生成。
"""

# 默认条件列：轮毂/轮缘半径、腹板角度、过渡半径、轴向范围等
CONDITION_COLUMNS = [
    "hub_r_end_mm",
    "rim_r_start_mm",
    "angle_web_deg",
    "r_trans_bore_web_mm",
    "z_min",
]
