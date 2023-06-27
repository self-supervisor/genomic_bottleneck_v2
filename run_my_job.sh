#!/bin/bash

# Declare the lists
seed_list=(1 2 3 4 5)
number_of_cell_types_list=(64 32 16 8 2)
env_name_list=("halfcheetah")

# Get the values based on the task id
seed=${seed_list[$1]}
number_of_cell_types=${number_of_cell_types_list[$2]}  # This index will change based on task id now
env_name=${env_name_list[0]}  # This index is always 0

# Run the Python script with these parameters
python training_torch.py --env_name $env_name --is_weight_sharing True --seed $seed --number_of_cell_types $number_of_cell_types
