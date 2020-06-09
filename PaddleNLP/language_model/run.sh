#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

function run_train() {
    echo "training"
    python train.py \
        --data_path data/simple-examples/data/ \
        --model_type small \
        --rnn_model "dynamic" \
        --use_gpu False \
        #--init_from_pretrain_model models/0/params
}

run_train
