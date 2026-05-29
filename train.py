#!/usr/bin/env python3
"""【已弃用】MLP 方案训练入口。

请改用统一 CLI::

    python run.py train --scheme mlp --dataset single|F404 [--fast] [--stage ae|diff|all]

保留本文件仅为兼容旧文档或脚本中的调用路径；直接执行将打印迁移提示并退出。
"""
import run  # noqa: F401  # 保留导入以便静态分析仍识别 run 模块

if __name__ == "__main__":
    raise SystemExit(
        "Use: python run.py train --scheme mlp --dataset single|F404 [--fast] [--stage ae|diff|all]"
    )
