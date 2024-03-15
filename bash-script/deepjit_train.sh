#!/bin/bash

cd ../Com

python main.py -train \
    -project $1_$2 \
    -train_data "$3/$1/commits/deepjit_$1_$2_train.pkl" \
    -dictionary_data "$3/$1/commits/$1_$2_train_dict.pkl" \
    -num_epochs $4 \
    -auc "/home/manh/Documents/JIT/CC2Vec-JIT/outputs/auc.csv" \
    -testing_time "/home/manh/Documents/JIT/CC2Vec-JIT/outputs/testing_time.csv" \
    -training_time "/home/manh/Documents/JIT/CC2Vec-JIT/outputs/training_time.csv" \
    -ram "/home/manh/Documents/JIT/CC2Vec-JIT/outputs/ram.csv" \
    -vram "/home/manh/Documents/JIT/CC2Vec-JIT/outputs/vram.csv" \