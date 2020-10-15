#!/bin/bash

for seed in {0..9..1}
do
    for ld in 0.005 1
    do
        for pe in 0 0.2 0.4
        do
            for d in {10..250..20}
            do
                for k in 3
                do
                    for method in 1
                    do
                        python synthetic.py $seed $ld $pe $d 300 40000 $k $method
                    done
                done
            done
        done
    done
done
