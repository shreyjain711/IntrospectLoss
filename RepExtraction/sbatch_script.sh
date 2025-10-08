#!/bin/bash
#SBATCH --job-name=data_gen
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:4
#SBATCH --mem=256G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --partition=general

eval "$(conda shell.bash hook)"
conda activate capstone
export PYTHONPATH=.

cd /home/shreyj/IntrospectLoss/RepExtraction

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python extractor.py 8
