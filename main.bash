#!/bin/bash

for exp in {1..2}
    do
	# from 0 up to and including 9 with steps 1
    for seed in {0..9..1} 
        do
        for d in {10..1010..200}
        do
			work=/scratch/kadioglu.b/
			cd $work
			sbatch --job-name=${algorithm:(-1)}.$seed.${N:(-1)} --output=z.out --error=z.err single.bash $algorithm $seed $N
            done
        done
    done
