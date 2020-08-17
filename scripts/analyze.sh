#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --array=0
#SBATCH --job-name=analyze
#SBATCH --output=analyze_%A_%a.out

module purge
module load cuda-10.1

python -u /misc/vlgscratch4/LakeGroup/emin/bold5000/ROIs/analyze.py 

echo "Done"
