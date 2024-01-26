#!/bin/bash

cd ~/GraVAC/gravac_py3/

# run models like resnet101, vgg16 or lstm
worldsize=8
# compression can be topK, dgc, redsync or randomK
# perform compression by given compressor to target CF if 1, allreduce uncompressed gradients if 0.
docompression=1
cf=0.1
dir='/'
for rank in $(seq 1 $worldsize)
do
  procrank=$(($rank-1))
  python3 -m run.baseline_training --log-dir=$dir --train-dir=$dir --master-addr='127.0.0.1' --master-port='29810' \
  ---world-size=$worldsize --train-bsz=32 --compression='randomK' --model-name='vgg16' --dataset='cifar100' --lr=0.01
  --weight-decay=5e-4 --do-compression=$docompression --compression-ratio=$cf &
done