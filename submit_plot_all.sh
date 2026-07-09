#!/bin/bash
#SBATCH --partition=all      # Using normal partition
#SBATCH --container=el9hw
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --job-name=PLOTALL
#SBATCH --output=./logs/log_%x.out
#SBATCH --error=./logs/log_%x.err
#SBATCH --time=1:00:00      # 2 days (maximum time limit for normal partition)

#  source my python env
module purge
source /home/twu27/.bashrc
source /home/twu27/python3/env/oceannet/bin/activate
module load Python/3.10.8-GCCcore-12.2.0
hostname
start_time=`date +%s`

python -u plot_LC_all.py

end_time=`date +%s`
echo execution time was `expr $end_time - $start_time` s.