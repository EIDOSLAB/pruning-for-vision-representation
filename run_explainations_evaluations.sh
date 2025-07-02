#!/bin/bash

i=0;
while [ $i -le $2 ]
do
    python src/explainations_evaluation_metrics.py --models-path ../models/  --output-dir ../output --data-path src/datasets/VOC2012/VOCdevkit/VOC2012/JPEGImages/ --model $1 --pruning-iteration $i 
    i=$((i + 1));
done