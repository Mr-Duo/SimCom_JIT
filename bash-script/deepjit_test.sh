#! /bin/bash

cd ../Com

python main.py -predict \
    -project $1_$2 \
    -predict_data "$4/$1/commits/$1_$3_dextend.pkl" \
    -dictionary_data "$4/$1/commits/$1_$2_dict.pkl" \
    -load_model "model/$1_$2/epoch_$5.pt"
