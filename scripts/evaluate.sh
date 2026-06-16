#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH -J evaluate
#SBATCH -o logs/evaluation/evaluate_readmit_%j.out
#SBATCH -e logs/evaluation/evaluate_readmit_%j.out

module load miniforge3/25.3.0-3
source ${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh
conda activate icd_gpu

export PYTHONPATH="$PWD/src:$PYTHONPATH"
python src/evaluate/evaluate.py
