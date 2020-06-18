#!/bin/bash

#SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --exclude=hpc1,hpc2,hpc3,hpc4,hpc5,hpc6,hpc7,hpc8,hpc9,vine3,vine4,vine6,vine11,vine12,lion17,rose1,rose2,rose3,rose4,rose7,rose8,rose9,lion3
#SBATCH --gres=gpu:4
##SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=2:00:00
#SBATCH --array=2-299
#SBATCH --job-name=train
#SBATCH --output=train_%A_%a.out

module purge
module load cuda-10.0
source /home/eo41/venv/bin/activate

python -u /misc/vlgscratch4/LakeGroup/emin/bold5000/ROIs/train_resnext.py --model-name 'resnext101_32x8d_wsl' --subject 'CSI1' #--freeze-trunk

echo "Done"
