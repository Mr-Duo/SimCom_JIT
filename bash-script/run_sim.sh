#! /bin/bash

cd Sim

python sim_model.py \
    -project gerrit \
    -train_data "/home/aiotlab3/RISE/Manh/SimCom_JIT/data/hand_crafted_features/gerrit/k_train.csv" \
    -test_data "/home/aiotlab3/RISE/Manh/SimCom_JIT/data/hand_crafted_features/gerrit/k_test.csv"