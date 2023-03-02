#!/bin/bash

dataset_name=$1
data_path=$2
likelihood_n=$3
epochs=$4
h_dim=$5
depth=$6
norm=$7

# for prod_space in "e2"
# do
#   echo "----- Training: sc -----"
#   python -m mt.examples.run --dataset=$dataset_name \
#                             --data=$data_path \
#                             --model=$prod_space \
#                             --fixed_curvature=True \
#                             --h_dim=$h_dim \
#                             --depth=$depth \
#                             --norm=$norm \
#                             --architecture="ff" \
#                             --likelihood_n=$likelihood_n \
#                             --epochs=$epochs \
#                             --warmup=100 \
#                             --lookahead=50 \
#                             --device="cuda:1"
# done

for prod_space in "e4"
do
  echo "----- Training: sc -----"
  python -m mt.examples.run --dataset=$dataset_name \
                            --data=$data_path \
                            --model=$prod_space \
                            --fixed_curvature=True \
                            --h_dim=$h_dim \
                            --depth=$depth \
                            --norm=$norm \
                            --architecture="ff" \
                            --likelihood_n=$likelihood_n \
                            --epochs=$epochs \
                            --warmup=150 \
                            --lookahead=50 \
                            --device="cuda:1"
done
