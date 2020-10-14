#!/bin/bash

for seed in {0..1..1}
do
    for ld in 0.005 1
    do
        for method in 1 2
        do
            python synthetic_by_M.py $seed $ld 2 500 1000 $method
        done
    done
done
