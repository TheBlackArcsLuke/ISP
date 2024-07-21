#!/bin/bash

#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2GB
#SBATCH --output=ISP_Output.txt

# Load modules 

module load python/3.11 scipy-stack

# Engage venv here

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Install

pip install --no-index --upgrade pip
pip install --no-index  scikit_learn

# Run scripts

python python food_filter.py
python ISP_script.py