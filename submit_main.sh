#!/bin/bash -l
#SBATCH --partition=ai2es      # Using normal partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=500M
#SBATCH --job-name=MAIN
#SBATCH --output=/ourdisk/hpc/ai2es/twu27/oceannet2/ens_pred_op/logs/log_%x.out
#SBATCH --error=/ourdisk/hpc/ai2es/twu27/oceannet2/ens_pred_op/logs/log_%x.err
#SBATCH --time=12:00:00      # 2 days (maximum time limit for normal partition)

hostname
start_time=`date +%s`

# Submit download 
cd /ourdisk/hpc/ai2es/twu27/srcdata/Mercator_hourly
jid1=$(sbatch --parsable submit_download_prep_mercator_nrt.sh)
echo "Submitted Mercator download with Job ID $jid1" | ts
sacct -j $jid1 --format=State --noheader | grep -q COMPLETED
while [ $? -ne 0 ]; do
    sleep 5
    sacct -j $jid1 --format=State --noheader | grep -q COMPLETED
done
echo "Mercator download completed." | ts

# Submit prediction
cd /ourdisk/hpc/ai2es/twu27/oceannet2/ens_pred_op
jid2=$(sbatch --parsable submit_prediction_120days.sh)
echo "Submitted prediction with Job ID $jid2" | ts
sacct -j $jid2 --format=State --noheader | grep -q COMPLETED
while [ $? -ne 0 ]; do
    sleep 5
    sacct -j $jid2 --format=State --noheader | grep -q COMPLETED
done
echo "Prediction completed." | ts

# Submit combine NetCDF
cd /ourdisk/hpc/ai2es/twu27/oceannet2/ens_pred_op
jid3=$(sbatch --parsable submit_combine_nc.sh)
echo "Submitted combine ensemble NetCDF with Job ID $jid3" | ts
sacct -j $jid3 --format=State --noheader | grep -q COMPLETED
while [ $? -ne 0 ]; do
    sleep 5
    sacct -j $jid3 --format=State --noheader | grep -q COMPLETED
done
echo "Combine ensemble NetCDF completed." | ts

# Submit plot
cd /ourdisk/hpc/ai2es/twu27/oceannet2/ens_pred_op
jid4=$(sbatch --parsable submit_plot_all.sh)
echo "Submitted plot with Job ID $jid4" | ts
sacct -j $jid4 --format=State --noheader | grep -q COMPLETED
while [ $? -ne 0 ]; do
    sleep 5
    sacct -j $jid4 --format=State --noheader | grep -q COMPLETED
done
echo "Plot completed." | ts

# Submit nc for visualization
cd /ourdisk/hpc/ai2es/twu27/oceannet2/ens_pred_op
jid5=$(sbatch --parsable submit_nc4vis.sh)
echo "Submitted make nc for visualization with Job ID $jid5" | ts
sacct -j $jid5 --format=State --noheader | grep -q COMPLETED
while [ $? -ne 0 ]; do
    sleep 5
    sacct -j $jid5 --format=State --noheader | grep -q COMPLETED
done
echo "nc4vis completed." | ts

end_time=`date +%s`
echo execution time was `expr $end_time - $start_time` s.