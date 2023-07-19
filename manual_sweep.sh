seed_list=(1 2 3 4 5)
number_of_cell_types_list=(64 32 16 4 2)
env_name_list=("ant")


for seed in ${seed_list[@]};
do
    for number_of_cell_types in ${number_of_cell_types_list[@]};
    do
        python training_torch.py --env_name ant --is_weight_sharing True --seed $seed --number_of_cell_types $number_of_cell_types
    done
done
