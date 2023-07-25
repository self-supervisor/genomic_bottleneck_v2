#!/bin/bash

# Declare the lists
seed_list=(1 2 3 4 5)
complexity_cost_list=(0.0001)
number_of_cell_types_list=(64 32 16 8 4 2)
num_timesteps=100000000
num_evals=30
reward_scaling=10
episode_length=1000
unroll_length=5
num_minibatches=32
num_updates_per_batch=4
discounting=0.97
learning_rate=3e-4
entropy_cost=1e-2
num_envs=4096
batch_size=2048

env_name="ant"

# The script now expects indices for seed, complexity_cost, and cell types
seed_idx=$1
complexity_cost_idx=$2
number_of_cell_types_idx=$3

# Fetch the actual values using the indices
seed=${seed_list[$seed_idx]}
complexity_cost=${complexity_cost_list[$complexity_cost_idx]}
number_of_cell_types=${number_of_cell_types_list[$number_of_cell_types_idx]}

# Run the Python script with these parameters
python training_torch.py --complexity_cost $complexity_cost --env_name $env_name --is_weight_sharing True --seed $seed --number_of_cell_types $number_of_cell_types --num_timesteps $num_timesteps --num_evals $num_evals --reward_scaling $reward_scaling --episode_length $episode_length --unroll_length $unroll_length --num_minibatches $num_minibatches --num_updates_per_batch $num_updates_per_batch --discounting $discounting --learning_rate $learning_rate --entropy_cost $entropy_cost --num_envs $num_envs --batch_size $batch_size
