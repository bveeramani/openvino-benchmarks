#!/bin/bash
TEMP_DIR=/tmp/models
INSTALL_DIR=/opt/intel/openvino
PROJECT_DIR=$(dirname "$0")/../..

mkdir -p $TEMP_DIR

caffe()
{
    MODEL_NAME=$1
    CAFFEMODEL_URL=$2
    PROTOTXT_URL=$3

    curl -o $TEMP_DIR/$MODEL_NAME.caffemodel $CAFFEMODEL_URL
    curl -o $TEMP_DIR/$MODEL_NAME.prototxt $PROTOTXT_URL
    python3 $INSTALL_DIR/deployment_tools/model_optimizer/mo.py \
        --input_model $TEMP_DIR/$MODEL_NAME.caffemodel \
        --input_proto $TEMP_DIR/$MODEL_NAME.prototxt \
        --output_dir $PROJECT_DIR/models/fp32 \
        --model_name $MODEL_NAME \
        --data_type FP32
}


caffe AlexNet \
    http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel \
    https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt
