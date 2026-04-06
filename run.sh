#!/bin/bash
#SBATCH --job-name=run
#SBATCH --output=logs/id_%j.out
#SBATCH --error=logs/id_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --qos=default
#SBATCH --partition=general

# 完整配置（如果需要编译CUDA程序）
export CUDA_HOME=/usr/local/cuda-13.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 禁用 Python stdout 缓冲，确保日志实时输出
export PYTHONUNBUFFERED=1

# 加速 CUDA JIT 编译（只编译 A800 的 sm_80 架构）
export TORCH_CUDA_ARCH_LIST="8.0"

# 无显示器环境下使用 offscreen 渲染
export QT_QPA_PLATFORM=offscreen

cd /home/shining/jhl/projects/Foundationpose_for_VR
pixi run python -u src/pipeline/quest_pipeline.py