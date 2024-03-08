#! /bin/bash

cd ../Com

python main.py -predict \
    -project $1_$2 \
    -do_valid \
    -predict_data "$5/$1/commits/simcom_$1_$3.pkl" \
    -dictionary_data "$4/$1/commits/$1_$2_train_dict.pkl" \
    -load_model "model/$1_$2/best_model.pt" \
    -auc "/home/manh/Documents/JIT/CC2Vec-JIT/outputs/auc.csv" \
    -testing_time "/home/manh/Documents/JIT/CC2Vec-JIT/outputs/testing_time.csv" \
    -training_time "/home/manh/Documents/JIT/CC2Vec-JIT/outputs/training_time.csv" \
    -ram "/home/manh/Documents/JIT/CC2Vec-JIT/outputs/ram.csv" \
    -vram "/home/manh/Documents/JIT/CC2Vec-JIT/outputs/vram.csv" \
