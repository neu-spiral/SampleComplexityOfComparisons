#!/bin/bash

for seed in {0..9..1}
do
    for ld in 0.005 1
    do
        for d in 20 100 250
        do
            for method in 1 2
            do
                python synthetic_by_M.py $seed $ld $d 2000 50000 $method
            done
        done
    done
done