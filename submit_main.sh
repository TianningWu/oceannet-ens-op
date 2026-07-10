#!/bin/bash -l
#SBATCH --partition=ai2es,all      # Using normal partition
#SBATCH --container=el9hw
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=500M
#SBATCH --job-name=MAIN
#SBATCH --output=/ourdisk/hpc/ai2es/twu27/oceannet2/ens_pred_op/logs/log_%x.out
#SBATCH --error=/ourdisk/hpc/ai2es/twu27/oceannet2/ens_pred_op/logs/log_%x.err
#SBATCH --time=20:00:00      # 2 days (maximum time limit for normal partition)

hostname
start_time=`date +%s`

mercator_dir=/ourdisk/hpc/ai2es/twu27/srcdata/Mercator_hourly
ens_pred_dir=/ourdisk/hpc/ai2es/twu27/oceannet2/ens_pred_op

# Submit download 
cd "$mercator_dir"
jid1=$(sbatch --parsable --chdir="$mercator_dir" submit_download_prep_mercator_nrt.sh)
echo "Submitted Mercator download with Job ID $jid1" | ts
sacct -j $jid1 --format=State --noheader | grep -q COMPLETED
while [ $? -ne 0 ]; do
    sleep 5
    sacct -j $jid1 --format=State --noheader | grep -q COMPLETED
done
echo "Mercator download completed." | ts

# Submit prediction
cd "$ens_pred_dir"
jid2=$(sbatch --parsable --chdir="$ens_pred_dir" submit_prediction_120days.sh)
echo "Submitted prediction with Job ID $jid2" | ts
sacct -j $jid2 --format=State --noheader | grep -q COMPLETED
while [ $? -ne 0 ]; do
    sleep 5
    sacct -j $jid2 --format=State --noheader | grep -q COMPLETED
done
echo "Prediction completed." | ts

# Submit combine NetCDF
cd "$ens_pred_dir"
jid3=$(sbatch --parsable --chdir="$ens_pred_dir" submit_combine_nc.sh)
echo "Submitted combine ensemble NetCDF with Job ID $jid3" | ts
sacct -j $jid3 --format=State --noheader | grep -q COMPLETED
while [ $? -ne 0 ]; do
    sleep 5
    sacct -j $jid3 --format=State --noheader | grep -q COMPLETED
done
echo "Combine ensemble NetCDF completed." | ts

# Submit plot
cd "$ens_pred_dir"
jid4=$(sbatch --parsable --chdir="$ens_pred_dir" submit_plot_all.sh)
echo "Submitted plot with Job ID $jid4" | ts
sacct -j $jid4 --format=State --noheader | grep -q COMPLETED
while [ $? -ne 0 ]; do
    sleep 5
    sacct -j $jid4 --format=State --noheader | grep -q COMPLETED
done
echo "Plot completed." | ts

# Submit nc for visualization
cd "$ens_pred_dir"
jid5=$(sbatch --parsable --chdir="$ens_pred_dir" submit_nc4vis.sh)
echo "Submitted make nc for visualization with Job ID $jid5" | ts
sacct -j $jid5 --format=State --noheader | grep -q COMPLETED
while [ $? -ne 0 ]; do
    sleep 5
    sacct -j $jid5 --format=State --noheader | grep -q COMPLETED
done
echo "nc4vis completed." | ts

end_time=`date +%s`
echo execution time was `expr $end_time - $start_time` s.
