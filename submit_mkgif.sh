#!/bin/bash -l
#SBATCH -p ai2es,all
#SBATCH --container=el9hw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --chdir=./
#SBATCH --job-name="MKGIF"
#SBATCH --output=./logs/log_%x.out
#SBATCH --error=./logs/log_%x.err
#SBATCH --mem=1G

#  source my python env
module purge
source /home/twu27/.bashrc
source /home/twu27/python3/env/oceannet/bin/activate
module load Python/3.10.8-GCCcore-12.2.0
hostname
start_time=`date +%s`

python -u mkvid_standalone.py

end_time=`date +%s`
echo execution time was `expr $end_time - $start_time` s.