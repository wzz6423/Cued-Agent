#!/bin/bash

# 检查是否在虚拟环境中
if [[ "$VIRTUAL_ENV" == "" && "$CONDA_DEFAULT_ENV" == "" ]]; then
    echo "警告: 未检测到激活的虚拟环境。"
    echo "建议先创建并激活一个虚拟环境 (例如: conda create -n cued_agent python=3.8 -y && conda activate cued_agent)"
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "正在安装基础依赖..."
pip install -r requirements.txt

echo "正在安装 Fairseq..."
cd lip_agent_and_prompt_decoding_agent/fairseq
pip install -e .
cd ../..

echo "环境配置完成！"
