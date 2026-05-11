#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition bigmem
#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH -J preprocessing
#SBATCH -o logs/preprocessing/preprocessing%j.out
#SBATCH -e logs/preprocessing/preprocessing%j.out

module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate icd_gpu

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python src/preprocessing.py
