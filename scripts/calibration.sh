#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH -J calibration
#SBATCH -o logs/calibration/calibration%j.out
#SBATCH -e logs/calibration/calibration%j.out

module load cuda
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate icd_gpu

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python src/calibration/calibration_curve.py
