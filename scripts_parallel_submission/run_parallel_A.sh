#!/bin/bash

#SBATCH --array=0-11                 # bR 0-16, NPacific 0-4
#SBATCH --partition=week             # Using 'hourly' will grant higher priority
#SBATCH --ntasks-per-core=1          # Force no Hyper-Threading, will run 1 task per core

##SBATCH --time=1-00:00:00           # Define max time job will run

WORKER_ID=$SLURM_ARRAY_TASK_ID       # From 0 to ...

python calculate_A_parallel.py --index $WORKER_ID