#!/bin/bash

#$ -cwd
#$ -j y
#$ -t 1-50
#$ -o cheetah_fix
#$ -e cheetah_fix
#$ -N cheetah_fix
#$ -l gpu=1

CONFIG_NAMES=("halfcheetah" "ant")  # Add as many config names as you have
SEEDS=(0 1 2 3 4)  # Adjust the seed range as you need
NUM_CELL_TYPES=(64 32 16 8 4)  # Adjust as per your requirement

# Calculate indices for array job
CONFIG_INDEX=$((($SGE_TASK_ID-1)/(${#SEEDS[@]}*${#NUM_CELL_TYPES[@]})))
SEED_INDEX=$((($SGE_TASK_ID-1)/${#NUM_CELL_TYPES[@]}%${#SEEDS[@]}))
CELL_INDEX=$((($SGE_TASK_ID-1)%${#NUM_CELL_TYPES[@]}))

# Extract actual values using indices
CONFIG_NAME=${CONFIG_NAMES[$CONFIG_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}
CELL_TYPE=${NUM_CELL_TYPES[$CELL_INDEX]}

# Now call your python script with these parameters
python training_torch.py config-name=$CONFIG_NAME seed=$SEED number_of_cell_types=$CELL_TYPE hidden_size=64
