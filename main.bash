#!/bin/bash

for seed in {0..4..1}
do
    for ld in 0.005 1
    do
        for d in {10..250..20}
        do
            for k in 1 3
            do
                for method in 1 2
                do
                    sbatch --job-name=$seed${ld:(-2)}$d$k$method --output=z.out --error=z.err single.bash $seed $ld $d 300 50000 $k $method
                done
            done
        done
    done
done
