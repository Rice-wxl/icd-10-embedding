#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition gpu # Partition (queue) name
#SBATCH --gres=gpu:1 # Request 1 gpu
#SBATCH --nodes=1 # Number of nodes
#SBATCH --mem=96G
#SBATCH -J run_small_verification_mor30_
#SBATCH -o logs/training/run_small_verification_mor30_%j.out
#SBATCH -e logs/training/run_small_verification_mor30_%j.out

# module load cuda
# module load miniconda3/23.11.0s
# source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

# conda activate icd

# python transformer.py


module purge
unset LD_LIBRARY_PATH
export APPTAINER_BINDPATH="/oscar/home/$USER,/oscar/scratch/$USER,/oscar/data"
cd "$(dirname "$0")/.."
srun apptainer exec --nv tensorflow-24.03-tf2-py3.simg python transformer.py