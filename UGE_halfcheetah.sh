#!/bin/bash

#$ -cwd
#$ -o kl_cheetah
#$ -e kl_cheetah
#$ -N kl_cheetah
#$ -t 1-30
#$ -pe threads 8
#$ -l gpu=1

# Call the shell script with these indices
let "seed_idx = (${SGE_TASK_ID}-1) / 6 % 5"
let "complexity_cost_idx = (${SGE_TASK_ID}-1) / 30 % 1"
let "number_of_cell_types_idx = (${SGE_TASK_ID}-1) % 6"

bash halfcheetah.sh $seed_idx $complexity_cost_idx $number_of_cell_types_idx
