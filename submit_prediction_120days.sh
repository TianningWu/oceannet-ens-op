#!/bin/bash -l
#SBATCH -p ai2es,sooner_gpu_test
#SBATCH --container=el9hw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=./
#SBATCH --job-name="FNO_CP"
#SBATCH --output=./logs/log_%x.out
#SBATCH --error=./logs/log_%x.err

#  source my python env
module purge
source /home/twu27/.bashrc
source /home/twu27/python3/env/oceannet/bin/activate
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/11.8.0
hostname
start_time=`date +%s`

python -u run_model_prediction_120days_ensemble.py

end_time=`date +%s`
echo execution time was `expr $end_time - $start_time` s.