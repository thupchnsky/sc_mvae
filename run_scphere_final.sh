#!/bin/bash


for prod_space in "s3" "s4"
do
  echo "----- Training: sc -----"
  python -m mt.examples.run --dataset="scphere" \
                            --data="/data/shared/vishal/scPhere/lung_human_ASK440.mtx" \
                            --label="/data/shared/vishal/scPhere/lung_human_ASK440_celltype.tsv" \
                            --model=$prod_space \
                            --fixed_curvature=False \
                            --h_dim=300 \
                            --depth=3 \
                            --norm="None" \
                            --architecture="ff" \
                            --likelihood_n=300 \
                            --epochs=300 \
                            --warmup=100 \
                            --lookahead=50 \
                            --device="cuda:1" \
                            --batch_size=128
                            
  python -m mt.examples.run --dataset="scphere" \
                            --data="/data/shared/vishal/scPhere/nk_human_one.mtx" \
                            --label="/data/shared/vishal/scPhere/nk_human_one_celltype.tsv" \
                            --model=$prod_space \
                            --fixed_curvature=False \
                            --h_dim=300 \
                            --depth=3 \
                            --norm="None" \
                            --architecture="ff" \
                            --likelihood_n=300 \
                            --epochs=300 \
                            --warmup=100 \
                            --lookahead=50 \
                            --device="cuda:1" \
                            --batch_size=128
done