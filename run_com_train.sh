#! /bin/bash

cd Com

python main.py -train \
    -project gerrit \
    -train_data "/home/aiotlab3/RISE/Manh/SimCom_JIT/data/commit_cotents/processed_data/gerrit/gerrit_train.pkl" \
    -dictionary_data "/home/aiotlab3/RISE/Manh/SimCom_JIT/data/commit_cotents/processed_data/gerrit/gerrit_dict.pkl"