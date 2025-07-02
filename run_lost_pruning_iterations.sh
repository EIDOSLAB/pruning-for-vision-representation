#!/bin/bash

i=0;
while [ $i -le $2 ]
do
    python main_lost.py --dataset VOC12 --set trainval --arch $1 --output_dir ../../lost --models_dir /home/cassano/models --pruning_iteration $i --resnet_dilate 1 --patch_size 32
    python main_lost.py --dataset VOC12 --set val --arch $1 --output_dir ../../lost --models_dir /home/cassano/models --pruning_iteration $i --resnet_dilate 1 --patch_size 32

    # python main_lost.py --dataset VOC12 --set trainval --arch $1 --output_dir ../../lost --models_dir /home/cassano/models --pruning_iteration $i --resnet_dilate 2
    # python main_lost.py --dataset VOC12 --set val --arch $1 --output_dir ../../lost --models_dir /home/cassano/models --pruning_iteration $i --resnet_dilate 2

    # python main_lost.py --dataset VOC12 --set trainval --arch $1 --output_dir ../../lost --models_dir /home/cassano/models --pruning_iteration $i --resnet_dilate 4
    # python main_lost.py --dataset VOC12 --set val --arch $1 --output_dir ../../lost --models_dir /home/cassano/models --pruning_iteration $i --resnet_dilate 4

    # python main_lost.py --dataset VOC12 --set trainval --arch $1 --output_dir ../../lost --models_dir /home/cassano/models --pruning_iteration $i --patch_size 4
    # python main_lost.py --dataset VOC12 --set val --arch $1 --output_dir ../../lost --models_dir /home/cassano/models --pruning_iteration $i --patch_size 4

    i=$((i + 1));

done
