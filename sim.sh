#! /bin/bash

cd Sim

python sim_model.py \
    -project openstack \
    -train_data "../data/hand_crafted_features/openstack/k_train.csv" \
    -test_data "../data/hand_crafted_features/openstack/k_test.csv"