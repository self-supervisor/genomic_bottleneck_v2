#!/bin/bash

#$ -cwd
#$ -o kl_ant
#$ -e kl_ant
#$ -N kl_ant
#$ -t 1-30
#$ -pe threads 8
#$ -l gpu=1

let "seed_idx = (${SGE_TASK_ID}-1) / (1*6) % 5"  # Distribute among the 5 seeds
let "complexity_cost_idx = (${SGE_TASK_ID}-1) / 6 % 1"  # Always 0 as there's only one value
let "number_of_cell_types_idx = (${SGE_TASK_ID}-1) % 6"  # Distribute among the 6 number_of_cell_types values

bash ant.sh $seed_idx $complexity_cost_idx $number_of_cell_types_idx
