#!/bin/bash

for seed in {0..9..1}
do
    for ld in 0.01 0.5 1
    do
        for d in {10..1010..200}
        do
            for k in 1 2 3 4
            do
                for method in 1 2
                do
                    sbatch --job-name=$seed$ld$d$k$method --output=z.out --error=z.err single.bash $seed $ld $d $d 1e4 $k $method
                done

            done
        done
    done
done
