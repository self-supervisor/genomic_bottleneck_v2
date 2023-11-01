#!/bin/bash

#$ -cwd
#$ -j y
#$ -t 1-50
#$ -o cheetah_and_ant
#$ -e cheetah_and_ant
#$ -N cheetah_and_ant
#$ -l gpu=1

CONFIG_NAMES=("halfcheetah" "ant")
SEEDS=(0 1 2 3 4)
NUM_CELL_TYPES=(128 64 48 32 16)

CONFIG_INDEX=$((($SGE_TASK_ID-1)/(${#SEEDS[@]}*${#NUM_CELL_TYPES[@]})))
SEED_INDEX=$((($SGE_TASK_ID-1)/${#NUM_CELL_TYPES[@]}%${#SEEDS[@]}))
CELL_INDEX=$((($SGE_TASK_ID-1)%${#NUM_CELL_TYPES[@]}))

CONFIG_NAME=${CONFIG_NAMES[$CONFIG_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}
CELL_TYPE=${NUM_CELL_TYPES[$CELL_INDEX]}

python training_torch.py --config-name=$CONFIG_NAME seed=$SEED number_of_cell_types=$CELL_TYPE hidden_size=128
