#!/usr/bin/env python3
"""PCA-UNet 方案训练与测试结果可视化 CLI。

与 ``visualize.py`` 类似，但针对 UNet 扩散架构与 PCA 潜空间，
入口实现位于 ``src.visualize.unet_results``。

典型用法::

    python visualize_unet.py --run-dir outputs/pca_unet/single/20250101_120000

具体参数见 ``src.visualize.unet_results.main`` 的 argparse 定义。
"""
from src.visualize.unet_results import main

if __name__ == "__main__":
    main()
