seed_list=(5)
learning_rate_list=(1e-2, 1e-3, 1e-4, 1e-5)
entropy_cost_list=(1e-2 1e-1 1e-3)
number_envs_list=(2048 1024)
batch_size_list=(1024 512 32)

for seed in ${seed_list[@]};
do
    for learning_rate in ${learning_rate_list[@]};
    do
        for entropy_cost in ${entropy_cost_list[@]};
        do
            for number_envs in ${number_envs_list[@]};
            do
                for batch_size in ${batch_size_list[@]};
                do
                    python training_torch.py --seed $seed --batch_size $batch_size --entropy_cost $entropy_cost --number_envs $number_envs --learning_rate $learning_rate
                done
            done
        done
    done
done
