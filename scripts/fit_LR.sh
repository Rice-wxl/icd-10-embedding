#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH -J fit_LR
#SBATCH -o logs/baselines/fit_LR%j.out
#SBATCH -e logs/baselines/fit_LR%j.out

module load cuda
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate icd_gpu

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python src/evaluate/fit_LR_baseline.py
