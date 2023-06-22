seed_list=(5)
number_of_cell_types_list=(64 32 16 4 2)
learning_rate_list=(3e-4)
entropy_cost_list=(1e-2)
batch_size_list=(1024)
env_name_list=("halfcheetah" "hopper" "walker2d")
# env_name_list=("ant")

for env_name in "${env_name_list[@]}";
do
    for seed in "${seed_list[@]}";
    do
        python training_torch.py --env_name $env_name --seed $seed --is_weight_sharing False
    done
done


for env_name in "${env_name_list[@]}";
do
    for seed in ${seed_list[@]};
    do
        for learning_rate in ${learning_rate_list[@]};
        do
            for entropy_cost in ${entropy_cost_list[@]};
            do
                for batch_size in ${batch_size_list[@]};
                do
                    for number_of_cell_types in ${number_of_cell_types_list[@]};
                    do
                        python training_torch.py --env_name $env_name --is_weight_sharing True --seed $seed --batch_size $batch_size --entropy_cost $entropy_cost --learning_rate $learning_rate --number_of_cell_types $number_of_cell_types
                    done
                done
            done
        done
    done
done
