#!/bin/bash

for seed in {0..9..1}
do
    for ld in 0.005 0.1 1
    do
        for pe in 0 0.2 0.4
        do
            for d in {10..250..20}
            do
                for k in 3
                do
                    for method in 1
                    do
                        sbatch --job-name=$seed${ld:(-2)}$d$k$method --output=z.out --error=z.err single.bash $seed $ld $pe $d 300 30000 $k $method
                    done
                done
            done
        done
    done
done
