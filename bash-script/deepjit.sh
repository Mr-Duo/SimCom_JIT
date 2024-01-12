#!/bin/bash

echo "Project: $1"
echo "Train: $2"
echo "Test: $3"
echo "Epochs: $6"
echo "Train dir: $4"
echo "Val dir: $5"

# bash deepjit_train.sh $1 $2 $4 $6

bash deepjit_test.sh $1 $2 $3 $4 $5 $6