#!/bin/bash

#SBATCH --nodes=1
##SBATCH --ntasks=1
##SBATCH --exclude=hpc1,hpc2,hpc3,hpc4,hpc5,hpc6,hpc7,hpc8,hpc9,vine4,vine12
##SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=6:00:00
#SBATCH --array=0
#SBATCH --job-name=split
#SBATCH --output=split_%A_%a.out


module purge
#module load cuda-10.0
source /home/eo41/venv/bin/activate


python -u /misc/vlgscratch4/LakeGroup/emin/bold5000/ROIs/split_train_test.py

echo "Done"
