#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition bigmem # Partition (queue) name
#SBATCH --nodes=1 # Number of nodes
#SBATCH --mem=256G
#SBATCH -J preprocessing_counting
#SBATCH -o logs/preprocessing/preprocessing_counting%j.out
#SBATCH -e logs/preprocessing/preprocessing_counting%j.out

module load cuda
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

conda activate icd

cd "$(dirname "$0")/.."
python preprocessing.py