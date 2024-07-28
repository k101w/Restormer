#!/bin/bash

#SBATCH --job-name=denoising
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --partition=all-shared-gpu-preemp
#SBATCH --gres=gpu:4
#SBATCH -c 48
module purge

module load openmpi
ulimit -l unlimited
echo "== This is the scripting step! =="

./train.sh Denoising/Options/RealDenoising_Restormer.yml
echo "== End of Job =="