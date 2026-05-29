#!/usr/bin/env python3
"""【已弃用】PCA-UNet 方案训练入口。

请改用统一 CLI::

    python run.py train --scheme pca_unet --dataset single|F404 [--fast] [--stage ae|unet|all]

其中 ``--stage unet`` 表示仅训练 UNet 扩散阶段（需已有自编码器 run 目录时可配合 ``--ae-run-dir``）。
直接执行本脚本将打印迁移提示并退出。
"""
if __name__ == "__main__":
    raise SystemExit(
        "Use: python run.py train --scheme pca_unet --dataset single|F404 [--fast] [--stage ae|unet|all]"
    )
