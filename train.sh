#!/bin/bash -eux
#SBATCH --job-name=training
#SBATCH --account=sci-rabl
#SBATCH --output=train.out
#SBATCH --partition=gpu 
#SBATCH --gpus=1 
#SBATCH --time=01:00:00

# uncomment to receive email when job fails
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=smilla.fox@student.hpi.de

uv run  main.py
