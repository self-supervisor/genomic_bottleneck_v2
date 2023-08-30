# #!/bin/bash

# # Declare the lists
# seed_list=(1 2 3 4 5)
# number_of_cell_types_list=(64 16 2)
# entropy_cost_list=(1e-2 1e-3 1e-4)
# learning_rate_list=(3e-2 3e-3 3e-4 3e-5)
# clipping_val_list=(3e0 3e-1 3e-2)
# unroll_length_list=(1 5 20)
# batch_size_list=(512 1024)
# num_minibatches_list=(4 8 16 32)
# num_update_epochs_list=(2 4 8)
# env_name_list=("ant")

# # The script now expects indices for seed and cell types
# seed_idx=$1
# number_of_cell_types_idx=$2

# # Fetch the actual values using the indices
# seed=${seed_list[$seed_idx]}
# number_of_cell_types=${number_of_cell_types_list[$number_of_cell_types_idx]}
# env_name=${env_name_list[0]}  # This index is always 0

# # Run the Python script with these parameters
# python training_torch.py --env_name $env_name --is_weight_sharing True --seed $seed --number_of_cell_types $number_of_cell_types


# #!/bin/bash

# # Declare the lists
# seed_list=(1 2 3 4 5)
# complexity_cost_list=(0.00000001 0.0000001 0.0000001 0.000001 0.00001 0.0001)
# number_of_cell_types_list=(64 16)  # Assuming a list for the example
# env_name_list=("halfcheetah")

# # The script now expects indices for seed, complexity_cost, and cell types
# seed_idx=$1
# complexity_cost_idx=$2
# number_of_cell_types_idx=$3

# # Fetch the actual values using the indices
# seed=${seed_list[$seed_idx]}
# complexity_cost=${complexity_cost_list[$complexity_cost_idx]}
# number_of_cell_types=${number_of_cell_types_list[$number_of_cell_types_idx]}
# env_name=${env_name_list[0]}  # This index is always 0

# # Run the Python script with these parameters
# python training_torch.py --complexity_cost $complexity_cost --env_name $env_name --is_weight_sharing True --seed $seed --number_of_cell_types $number_of_cell_types
# Define lists
seed_list=(1 2 3 4 5)
number_of_cell_types_list=(64 32 16 8 4 2)

# The script now expects the task index
task_idx=$1

# Calculate seed and number_of_cell_types indices based on task index
let "seed_idx = (${task_idx}-1) / 6"  # 6 is the number of different cell types
let "number_of_cell_types_idx = (${task_idx}-1) % 6"

# Fetch the actual values using the indices
seed=${seed_list[$seed_idx]}
number_of_cell_types=${number_of_cell_types_list[$number_of_cell_types_idx]}

# Run the command
python training_torch.py --complexity_cost 0.0001 --env_name halfcheetah --is_weight_sharing True --seed $seed --number_of_cell_types $number_of_cell_types
