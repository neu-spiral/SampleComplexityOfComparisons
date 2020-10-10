#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=16Gb
#SBATCH --time=0-08:00:00
#SBATCH --partition=short

srun python main_synthetic.py $1 $2 $3 $4 $5 $6 $7
