#!/bin/bash

dataset_name=$1
data_path=$2
label_path=$3
# likelihood_n=$4
# epochs=$5
# h_dim=$6
# depth=$7
# norm=$8

for prod_space in "e4" "e6"
do
  echo "----- Training: sc -----"
  python -m mt.examples.run --dataset=$dataset_name \
                            --data=$data_path \
                            --label=$label_path \
                            --model=$prod_space \
                            --fixed_curvature=True \
                            --h_dim=300 \
                            --depth=3 \
                            --norm="None" \
                            --architecture="ff" \
                            --likelihood_n=300 \
                            --epochs=200 \
                            --warmup=100 \
                            --lookahead=50 \
                            --device="cuda:1" \
                            --batch_size=128
done