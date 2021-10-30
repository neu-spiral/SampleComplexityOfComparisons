#!/bin/bash

for seed in {0..9..1}
do
    for ld in 0.005 0.08791667 0.17083333 0.25375 0.33666667 0.41958333 0.5025 0.58541667 0.66833333 0.75125 0.83416667 0.91708333 1
    do
        for pe in 0 0.4
        do
            for d in 10 100
            do
                for k in 3
                do
                    for method in 1 3
                    do
                        sbatch --job-name=$seed${ld:(-2)}$d$k$method --output=z.out --error=z.err single.bash $seed $ld $pe $d 200 2000 $k $method
                    done
                done
            done
        done
    done
done
