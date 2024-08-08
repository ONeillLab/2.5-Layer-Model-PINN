#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --account=def-oneill
#SBATCH --nodes=3
#SBATCH --cpus-per-task=80
#SBATCH --job-name run_sim

module load NiaEnv/2022a python/3.11.5
module load netcdf/4.6.3


source ~/.virtualenvs/SimEnv/bin/activate


python3 PINN.py

deactivate