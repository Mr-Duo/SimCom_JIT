#! /bin/bash

cd Com

python main.py -predict \
    -project gerrit \
    -predict_data "/home/aiotlab3/RISE/Manh/data/gerrit/deepjit/gerrit_test_raw.pkl" \
    -dictionary_data "/home/aiotlab3/RISE/Manh/data/gerrit/deepjit/gerrit_dict.pkl" \
    -load_model "/home/aiotlab3/RISE/Manh/SimCom_JIT/Com/snapshot/2023-05-29_10-16-48/epoch_1.pt"