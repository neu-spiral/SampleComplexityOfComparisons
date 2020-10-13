#!/bin/bash

for seed in {0..2..1}
do
    for ld in 0.005 1
    do
        for d in {20..170..50}
        do
            for k in 1 3
            do
                for method in 1 2
                do
                    python synthetic.py $seed $ld $d 300 4000 $k $method
                done
            done
        done
    done
done
