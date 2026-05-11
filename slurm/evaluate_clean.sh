#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --partition gpu # Partition (queue) name
#SBATCH --gres=gpu:1 # Request 1 gpu
#SBATCH --nodes=1 # Number of nodes
#SBATCH --mem=96G
#SBATCH -J evaluate_newci_readmit_transformer
#SBATCH -o logs/evaluation/evaluate_newci_readmit_transformer%j.out
#SBATCH -e logs/evaluation/evaluate_newci_readmit_transformer%j.out


module load cuda
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

# conda activate icd
conda activate icd_gpu

cd "$(dirname "$0")/.."
python evaluate_clean.py

# module purge
# unset LD_LIBRARY_PATH
# export APPTAINER_BINDPATH="/oscar/home/$USER,/oscar/scratch/$USER,/oscar/data"
# srun apptainer exec --nv tensorflow-24.03-tf2-py3.simg python evaluate.py