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
while true; do
    ./train.sh Denoising/Options/RealDenoising_Restormer.yml --shm-size 100g
    if [$? -eq 0]; then
     echo 'training complete'
     break
    else
     echo 'failed and going to restart'
     sleep 10
    fi
done
echo "== End of Job =="