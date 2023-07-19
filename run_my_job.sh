#!/bin/bash

# Declare the lists
seed_list=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
number_of_cell_types_list=(64 4 3 2)
env_name_list=("ant")

# The script now expects indices for seed and cell types
seed_idx=$1
number_of_cell_types_idx=$2

# Fetch the actual values using the indices
seed=${seed_list[$seed_idx]}
number_of_cell_types=${number_of_cell_types_list[$number_of_cell_types_idx]}
env_name=${env_name_list[0]}  # This index is always 0

# Run the Python script with these parameters
python training_torch.py --env_name $env_name --is_weight_sharing True --seed $seed --number_of_cell_types $number_of_cell_types
