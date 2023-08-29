#!/bin/bash

#$ -cwd
#$ -o cheetah_fix
#$ -e cheetah_fix
#$ -N cheetah_fix
#$ -t 1-30
#$ -pe threads 8
#$ -l gpu=1

bash run_my_job.sh $SGE_TASK_ID
