#!/bin/bash

for seed in {0..9..1}
do
    for ld in 0.005 1
    do
        for d in 20 100 250
        do
            for method in 1 2
            do
                sbatch --job-name=$seed${ld:(-2)}$d$k$method --output=z.out --error=z.err single_by_M.bash $seed $ld $d 2000 50000 $method
            done
        done
    done
done
