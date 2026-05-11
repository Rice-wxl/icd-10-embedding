#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition batch # Partition (queue) name
#SBATCH --nodes=1 # Number of nodes
#SBATCH --mem=96G
#SBATCH -J interpretation_IG_readmit
#SBATCH -o logs/feature_importance/interpretation_IG_readmit%j.out
#SBATCH -e logs/feature_importance/interpretation_IG_readmit%j.out


module load cuda
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

# conda activate icd
conda activate icd_gpu

cd "$(dirname "$0")/.."
python IG.py

# module purge
# unset LD_LIBRARY_PATH
# export APPTAINER_BINDPATH="/oscar/home/$USER,/oscar/scratch/$USER,/oscar/data"
# srun apptainer exec --nv tensorflow-24.03-tf2-py3.simg python evaluate.py