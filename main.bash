#!/bin/bash

for seed in {0..9..1}
do
    for ld in 0.005 1
    do
        for pe in 0 0.4
        do
            for d in 10 250
            do
                for k in 3
                do
                    for method in 1 3
                    do
                        sbatch --job-name=$seed${ld:(-2)}$d$k$method --output=z.out --error=z.err single.bash $seed $ld $pe $d 300 30000 $k $method
                        sbatch --job-name=$seed${ld:(-2)}$d$k$method --output=zM.out --error=zM.err single_by_M.bash $seed $ld $pe $d 1000 30000 $method
                    done
                done
            done
        done
    done
done
