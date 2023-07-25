#!/bin/bash

#$ -cwd
#$ -o kl_ant
#$ -e kl_ant
#$ -N kl_ant
#$ -t 1-30
#$ -pe threads 8
#$ -l gpu=1

# Calculate indices for each list
let "seed_idx = (${SGE_TASK_ID}-1) / 6 % 5"          # 5 is the length of seed_list
let "complexity_cost_idx = (${SGE_TASK_ID}-1) / 30"  # 1 is the length of complexity_cost_list, but we use the full length of the array job to cycle through it
let "number_of_cell_types_idx = (${SGE_TASK_ID}-1) % 6" # 6 is the length of number_of_cell_types_list

# Call the shell script with these indices
bash ant.sh $seed_idx $complexity_cost_idx $number_of_cell_types_idx
