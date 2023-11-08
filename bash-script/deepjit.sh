#!/bin/bash

echo "Project: $1"
echo "Train: $2"
echo "Test: $3"
echo "Epochs: $5"
echo "Data dir: $4"

bash deepjit_train.sh $1 $2 $4 $5

bash deepjit_test.sh $1 $2 $3 $4 $5