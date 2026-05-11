#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --partition gpu # Partition (queue) name
#SBATCH --gres=gpu:1 # Request 1 gpu
#SBATCH --nodes=1 # Number of nodes
#SBATCH --mem=96G
#SBATCH -J delong_mort_nodie
#SBATCH -o logs/delong/delong_mort_nodie%j.out
#SBATCH -e logs/delong/delong_mort_nodie%j.out


module load cuda
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

conda activate icd_gpu

cd "$(dirname "$0")/.."
python delong_test.py