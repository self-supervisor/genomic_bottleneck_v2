#!/bin/bash

# Declare the lists
seed_list=(1 2 3 4 5)
number_of_cell_types_list=(2)  # This list now only contains one element
env_name_list=("halfcheetah")

# Get the values based on the task id
seed=${seed_list[$1]}
number_of_cell_types=${number_of_cell_types_list[0]}  # This index is always 0 now
env_name=${env_name_list[0]}  # This index is always 0 now

# Run the Python script with these parameters
python training_torch_cheetah.py --env_name $env_name --is_weight_sharing False --seed $seed --number_of_cell_types $number_of_cell_types
