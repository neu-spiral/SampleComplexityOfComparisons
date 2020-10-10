#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=32Gb
#SBATCH --time=0-06:00:00
#SBATCH --partition=large

srun python main_synthetic.py $1 $2 $3 $4 $5 $6 $7
