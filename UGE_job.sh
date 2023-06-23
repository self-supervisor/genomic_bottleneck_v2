#!/bin/bash

#$ -cwd
#$ -o elzar-logs
#$ -e elzar-logs
#$ -j y
#$ -N failed_seeds
#$ -t 1-2
#$ -pe threads 8
#$ -l gpu=1

# Calculate index for seed list (only index that changes now)
let "seed_idx = (${SGE_TASK_ID}-1) % 2"   # 2 is the length of seed_list

# Call the shell script with this index
bash run_my_job.sh $seed_idx
