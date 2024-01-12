#! /bin/bash

cd ../Com

python main.py -predict \
    -project $1_$2 \
    -predict_data "$4/$1/commits/deepjit_$1_$3.pkl" \
    -dictionary_data "$4/$1/commits/$1_$2_train_dict.pkl" \
    -load_model "model/$1_$2/best_model.pt" \
    -num_epochs $5
