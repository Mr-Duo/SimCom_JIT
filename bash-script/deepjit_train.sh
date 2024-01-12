#!/bin/bash

cd ../Com

python main.py -train \
    -project $1_$2 \
    -train_data "$3/$1/commits/deepjit_$1_$2_train.pkl" \
    -dictionary_data "$3/$1/commits/$1_$2_dict_train.pkl" \
    -num_epochs $4