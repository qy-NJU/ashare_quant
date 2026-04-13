#!/bin/bash

# 使用 Python 3.12 环境运行量化策略的脚本

# 激活 conda 环境
source /Users/qianye/anaconda3/etc/profile.d/conda.sh
conda activate ashare_quant_py312

# 运行指定的 Python 脚本
if [ $# -eq 0 ]; then
    echo "用法：./run_with_py312.sh <script.py> [args...]"
    echo ""
    echo "示例:"
    echo "  ./run_with_py312.sh runner.py configs/pipeline_config.yaml"
    echo "  ./run_with_py312.sh scripts/sync_data.py --limit 100"
    exit 1
fi

# 执行 Python 脚本
python "$@"

# 退出 conda 环境
conda deactivate
