#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=32Gb
#SBATCH --time=1-00:00:00
#SBATCH --partition=short

srun python synthetic_by_M.py $1 $2 $3 $4 $5 $6 $7
