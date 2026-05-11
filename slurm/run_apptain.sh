#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --partition gpu # Partition (queue) name
#SBATCH --gres=gpu:1 # Request 1 gpu
#SBATCH --nodes=1 # Number of nodes
#SBATCH --mem=96G
#SBATCH -J hypertune_readmit_auc
#SBATCH -o logs/training/hypertune_readmit_auc%j.out
#SBATCH -e logs/training/hypertune_readmit_auc%j.out

module purge
unset LD_LIBRARY_PATH
export APPTAINER_BINDPATH="/oscar/home/$USER,/oscar/scratch/$USER,/oscar/data"
cd "$(dirname "$0")/.."
srun apptainer exec --nv tensorflow-24.03-tf2-py3.simg python hyper_tune.py