#!/bin/bash
DEVICE=$1

ROOT=$(dirname "$0")/../..

if [[ "$DEVICE" == 'MYRIAD' ]]
then
    PRECISION=fp16
elif [[ "$DEVICE" == 'CPU' ]]
then
    PRECISION=fp32
else
    echo Expected CPU or Myriad as an argument.
    exit 9
fi

for model in $ROOT/models/$PRECISION/*.xml
do
    for api in sync async
    do
        python3 $ROOT/scripts/python/benchmark.py --model $model -d $DEVICE \
        --batch_size 32 -f ./benchmarks.csv --num_infer_requests 2 --api $api
    done
done
