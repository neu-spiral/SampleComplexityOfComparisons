#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=64Gb
#SBATCH --time=1-00:00:00
#SBATCH --partition=short

srun python main_synthetic.py $1 $2 $3 $4 $5 $6 $7