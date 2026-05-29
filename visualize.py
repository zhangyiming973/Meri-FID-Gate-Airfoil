#!/usr/bin/env python3
"""MLP 方案训练与测试结果可视化 CLI。

从 ``outputs/`` 下已保存的 run 目录读取 checkpoint、日志与生成样本，
调用 ``src.visualize.results`` 中的绘图与对比逻辑（重建误差、潜空间、扩散采样等）。

典型用法::

    python visualize.py --run-dir outputs/mlp/single/20250101_120000

具体参数见 ``src.visualize.results.main`` 的 argparse 定义。
"""
from src.visualize.results import main

if __name__ == "__main__":
    main()
