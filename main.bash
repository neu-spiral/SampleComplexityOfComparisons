#!/bin/bash

for seed in {0..2..1}
do
    for ld in 0.01 0.1 1
    do
        for d in {20..520..500}
        do
            for k in 1 3
            do
                for method in 1 2
                do
                    sbatch --job-name=$seed${ld:(-2)}$d$k$method --output=z.out --error=z.err single.bash $seed $ld $d $d 100000 $k $method
                done
            done
        done
    done
done
