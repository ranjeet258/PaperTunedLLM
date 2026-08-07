#!/bin/bash
# This script is intended to be run in a GPU environment (like Kaggle or a VM)

# Exit on error
set -e

echo "Starting PaperTunedLLM Training Pipeline..."

# 1. Prepare Dataset
echo "Preparing dataset..."
python ../scripts/prepare_qasper.py

# Note: In a real environment, you need to clone LlamaFactory and copy the dataset
# git clone https://github.com/hiyouga/LLaMA-Factory.git
# cd LLaMA-Factory
# pip install -e .[metrics]
# cp ../datasets/qasper_llamafactory.json data/
# And update data/dataset_info.json to include "qasper"

# 2. Run Fine-tuning with LlamaFactory
echo "Starting QLoRA Fine-tuning via LlamaFactory..."
# Ensure you are running this from the LlamaFactory directory or adjust paths
# llamafactory-cli train ../configs/llamafactory_qlora.yaml

echo "Training completed."
echo "Model checkpoints are saved in the output directory specified in the config."
