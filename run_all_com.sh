#!/bin/bash
cd "Com" || exit 1 # Exit if changing directory fails
for prj in go gerrit platform jdt openstack qt ;
do 
    python main.py -train \
        -project "$prj" \
        -do_valid \
        -train_data "/kaggle/input/splited-sim-com/splitted_data/$prj/partition=0.5/${prj}_train_part_1.pkl" \
        -test_data "/kaggle/input/splited-sim-com/data/commit_cotents/processed_data/$prj/val_train/${prj}_val.pkl" \
        -dictionary_data "/kaggle/input/splited-sim-com/splitted_data/$prj/partition=0.5/${prj}_dict_part_1.pkl" \
        -num_epochs 30

    python main.py -predict \
        -project "$prj" \
        -predict_data "/kaggle/input/splited-sim-com/data/commit_cotents/processed_data/$prj/${prj}_test.pkl" \
        -dictionary_data "/kaggle/input/splited-sim-com/splitted_data/$prj/partition=0.5/${prj}_dict_part_1.pkl" \
        -load_model "/kaggle/working/SimCom_JIT/Com/model/$prj/best_model.pt"
done
