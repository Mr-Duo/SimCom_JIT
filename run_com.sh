#!/bin/bash

echo "Project: $1"
echo "Train: $2"
echo "Test: $3"
echo "Epochs: $5"
echo "Data dir: $4"

bash com_train.sh $1 $2 $4 $5

bash com_test.sh $1 $2 $3 $4 $5