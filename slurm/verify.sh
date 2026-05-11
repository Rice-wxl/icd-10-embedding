#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --partition gpu # Partition (queue) name
#SBATCH --gres=gpu:1 # Request 1 gpu
#SBATCH --nodes=1 # Number of nodes
#SBATCH --mem=96G
#SBATCH -J verify
#SBATCH -o logs/verification/verify%j.out
#SBATCH -e logs/verification/verify%j.out

module load miniforge3/25.3.0-3
source ${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh

conda activate icd_gpu

cd "$(dirname "$0")/.."
python verify_LR_baseline.py

# module purge
# unset LD_LIBRARY_PATH
# export APPTAINER_BINDPATH="/oscar/home/$USER,/oscar/scratch/$USER,/oscar/data"
# srun apptainer exec --nv tensorflow-24.03-tf2-py3.simg python evaluate.py