#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --partition gpu # Partition (queue) name
#SBATCH --gres=gpu:1 # Request 1 gpu
#SBATCH --nodes=1 # Number of nodes
#SBATCH --mem=96G
#SBATCH -J calibration_mort_nodie_baseline
#SBATCH -o logs/calibration/calibration_mort_nodie_baseline%j.out
#SBATCH -e logs/calibration/calibration_mort_nodie_baseline%j.out


# module load cuda
# module load miniconda3/23.11.0s
# source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
module load miniforge3/25.3.0-3
source ${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh

conda activate icd_gpu

cd "$(dirname "$0")/.."
python calibration_curve.py
