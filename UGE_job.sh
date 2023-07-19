#!/bin/bash

#$ -cwd
#$ -o elzar-logs_cheetah_sweep
#$ -e elzar-logs_cheetah_sweep
#$ -N cheetah_sweep
#$ -t 1-60
#$ -tc 10
#$ -pe threads 8
#$ -l gpu=1

# Calculate indices for each list
let "seed_idx = (${SGE_TASK_ID}-1) / 4 % 15"  # 15 is the length of seed_list
let "cell_types_idx = (${SGE_TASK_ID}-1) % 4"  # 4 is the length of number_of_cell_types_list

# Call the shell script with these indices
bash run_my_job.sh $seed_idx $cell_types_idx
