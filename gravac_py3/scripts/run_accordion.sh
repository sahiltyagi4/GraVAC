#!/bin/bash

cd ~/GraVAC/gravac_py3/

# run models like resnet101, vgg16 or lstm
# compression can be topK, dgc, redsync or randomK
worldsize=8
dir='/'
for rank in $(seq 1 $worldsize)
do
  procrank=$(($rank-1))
  python3 -m run.accordion --log-dir=$dir --train-dir=$dir --master-addr='127.0.0.1' --master-port='29810' \
  ---world-size=$worldsize --train-bsz=32 --compression='randomK' --model-name='resnet101' --dataset='cifar10' --lr=0.01 \
  --weight-decay=5e-4 --low-cf=0.01 --high-cf=0.001 --norm-threshold=0.2 --dist-backend='gloo' --test-dir=$dir --rank=$procrank &
done