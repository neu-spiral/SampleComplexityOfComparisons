#!/bin/bash

for seed in {0..9..1}
do
    for ld in 0.005 1
    do
        for method in 1 2
        do
            sbatch --job-name=$seed${ld:(-2)}$d$k$method --output=z.out --error=z.err single.bash $seed $ld 200 2000 50000 $method
        done
    done
done
