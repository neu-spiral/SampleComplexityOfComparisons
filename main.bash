#!/bin/bash
# 3x6x3x2 = 108

for seed in {0..4..1}
do
    for ld in 0.01 0.1 1
    do
        for d in {20..1020..200}
        do
            for k in 1 2 3
            do
                for method in 1 2
                do
                    sbatch --job-name=$seed${ld:(-2)}$d$k$method --output=z.out --error=z.err single.bash $seed $ld $d $d 10000 $k $method
                done

            done
        done
    done
done