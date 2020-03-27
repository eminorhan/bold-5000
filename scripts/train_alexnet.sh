#!/bin/bash

#SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --exclude=hpc1,hpc2,hpc3,hpc4,hpc5,hpc6,hpc7,hpc8,hpc9,vine3,vine4,vine6,vine11,vine12,lion17,rose7,rose8,rose9
#SBATCH --gres=gpu:2
##SBATCH --cpus-per-task=1
#SBATCH --mem=100GB
#SBATCH --time=4:00:00
#SBATCH --array=60-79
#SBATCH --job-name=train
#SBATCH --output=train_%A_%a.out

module purge
module load cuda-10.0
source /home/eo41/venv/bin/activate

python -u /misc/vlgscratch4/LakeGroup/emin/bold5000/ROIs/train_alexnet.py --model-name 'alexnet_rand' --subject 'CSI1' #--freeze-trunk

echo "Done"
