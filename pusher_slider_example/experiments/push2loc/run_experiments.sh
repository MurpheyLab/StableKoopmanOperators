#!/bin/zsh

operators=("least squares" "stable")
methods=("finite horizon" "infinite horizon")
seeds=(0 1 2 3 4)

rm data/*.pkl

for op in $operators
do
    for me in $methods
    do
        for ss in {1..10}
        do
            python3 main.py --method $me --operator $op --seed $ss
            echo "seed: $s method: $me operator: $op"
        done
    done
done
