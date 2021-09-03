#!/bin/zsh

trials=100
for i in {1..$trials}
do
    python3 prediction_example.py
    echo "trials $i out of $trials"
done
