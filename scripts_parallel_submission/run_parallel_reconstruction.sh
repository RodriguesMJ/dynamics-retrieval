#!/bin/bash

#SBATCH --ntasks-per-core=1 

module purge
module load anaconda
conda activate myenv_nlsa

WORKER_ID=$SLURM_ARRAY_TASK_ID   # From 0 to ... 

# pass sbatch arguments to python
python -m nlsa.reconstruct_parallel $WORKER_ID "$@"