#!/bin/bash

cd ~/GraVAC/gravac_py3/

# run models like resnet101, vgg16 or lstm
worldsize=8
cf=0.1
dir='/'
compressor='gravacTopK'
for rank in $(seq 1 $worldsize)
do
  procrank=$(($rank-1))
  python3 -m run.compression_throughput --log-dir=$dir --train-dir=$dir --master-addr='127.0.0.1' --master-port='29810' \
  ---world-size=$worldsize --train-bsz=32 --compression=$compressor --model-name='resnet101' --dataset='cifar10' --lr=0.1 \
  --weight-decay=5e-4 &
done