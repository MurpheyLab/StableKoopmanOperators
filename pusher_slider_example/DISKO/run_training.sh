#!/bin/zsh

python3 train.py --X ../data/PsiX.npy --Y ../data/PsiY.npy --U ../data/U.npy --save_dir . --seed 2020 --store_matrix --step_size_factor 2 --eps 1e-24 --fgm_max_iter 1000
