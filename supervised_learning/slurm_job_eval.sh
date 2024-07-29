#!/bin/bash
#SBATCH --job-name=MNIST_10_seeds
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err
#SBATCH --array=0-769
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Load any necessary modules here
module load miniconda/3
conda activate py38

# Run the Python script
python eval_script.py --array_id $SLURM_ARRAY_TASK_ID
