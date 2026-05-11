#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH -J hyper_tune
#SBATCH -o logs/training/hyper_tune%j.out
#SBATCH -e logs/training/hyper_tune%j.out

module load cuda
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate icd_gpu

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python src/train/hyper_tune.py
