#!/bin/bash

for seed in {0..9..1}
do
    for ld in 0.005
    do
        for pe in 0 0.2 0.4
        do
            for d in 10 90 250
            do
                for k in 3
                do
                    for method in 1
                    do
                        python synthetic.py $seed $ld $pe $d 300 30000 $k $method
                        python synthetic_by_M.py $seed $ld $pe $d 2500 50000 $method
                    done
                done
            done
        done
    done
done
