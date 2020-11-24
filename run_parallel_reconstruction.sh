#!/bin/bash

#SBATCH --array=0-10                 # 0-9   bR 0-16, NPacific 0-8
#SBATCH --partition=day      ###rhel77_test              # Using 'hourly' will grant higher priority
#SBATCH --ntasks-per-core=1          # Force no Hyper-Threading, will run 1 task per core

##SBATCH --time=1-00:00:00           # Define max time job will run

WORKER_ID=$SLURM_ARRAY_TASK_ID       # From 0 to ...

python reconstruct_parallel.py --index $WORKER_ID