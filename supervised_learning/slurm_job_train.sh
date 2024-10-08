#!/bin/bash
#SBATCH --job-name=MNIST_10_seeds
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err
#SBATCH --array=0-69%10
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Load any necessary modules here
module load miniconda/3
conda activate py38

# Run the Python script
python Final_MNIST_SGB.py --array_id $SLURM_ARRAY_TASK_ID --data_path "data/" --save_path "/network/scratch/a/augustine.mavor-parker/MNIST_results/"  
