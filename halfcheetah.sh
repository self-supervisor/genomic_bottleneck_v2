#!/bin/bash

# Declare the lists
seed_list=(1 2 3 4 5)  # Five seeds
complexity_cost_list=(0.0001)  # Assuming you're still using this value
number_of_cell_types_list=(64 32 16 8 4 2)  # From previous example
env_name_list=("halfcheetah")

# Hyperparameters for halfcheetah
num_timesteps=300000000
eval_frequency=100
reward_scaling=1
episode_length=1000
unroll_length=20
num_minibatches=32
num_updates_per_batch=8
discounting=0.95
learning_rate=3e-4
entropy_cost=0.001
num_envs=2048
batch_size=512

# The script expects indices for seed, complexity_cost, and number_of_cell_types
seed_idx=$1
complexity_cost_idx=$2
number_of_cell_types_idx=$3

# Fetch the actual values using the indices
seed=${seed_list[$seed_idx]}
complexity_cost=${complexity_cost_list[$complexity_cost_idx]}
number_of_cell_types=${number_of_cell_types_list[$number_of_cell_types_idx]}
env_name=${env_name_list[0]}  # This index is always 0

# Run the Python script with these parameters
python training_torch.py --env_name $env_name --seed $seed --complexity_cost $complexity_cost --number_of_cell_types $number_of_cell_types --num_timesteps $num_timesteps --eval_frequency $eval_frequency --reward_scaling $reward_scaling --episode_length $episode_length --unroll_length $unroll_length --num_minibatches $num_minibatches --num_updates_per_batch $num_updates_per_batch --discounting $discounting --learning_rate $learning_rate --entropy_cost $entropy_cost --num_envs $num_envs --batch_size $batch_size
