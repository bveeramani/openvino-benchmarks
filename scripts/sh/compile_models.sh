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

    if [ ! -f "$TEMP_DIR/$MODEL_NAME.caffemodel" ];
    then
        curl -o $TEMP_DIR/$MODEL_NAME.caffemodel $CAFFEMODEL_URL
    fi

    if [ ! -f "$TEMP_DIR/$MODEL_NAME.prototxt" ];
    then
        curl -o $TEMP_DIR/$MODEL_NAME.prototxt $PROTOTXT_URL
    fi

    if [ ! -f "$PROJECT_DIR/models/fp32/$MODEL_NAME.xml" ];
    then
        python3 $INSTALL_DIR/deployment_tools/model_optimizer/mo.py \
            --input_model $TEMP_DIR/$MODEL_NAME.caffemodel \
            --input_proto $TEMP_DIR/$MODEL_NAME.prototxt \
            --output_dir $PROJECT_DIR/models/fp32 \
            --model_name $MODEL_NAME \
            --data_type FP32
    fi
}

onnx()
{
    MODEL_NAME=$1
    ONNX_URL=$2

    if [ ! -f "$TEMP_DIR/$MODEL_NAME.onnx" ];
    then
        curl -o $TEMP_DIR/$MODEL_NAME.onnx $ONNX_URL
    fi

    if [ ! -f "$PROJECT_DIR/models/fp32/$MODEL_NAME.xml" ];
    then
        python3 $INSTALL_DIR/deployment_tools/model_optimizer/mo.py \
            --input_model $TEMP_DIR/$MODEL_NAME.onnx \
            --output_dir $PROJECT_DIR/models/fp32 \
            --model_name $MODEL_NAME \
            --data_type FP32
    fi
}


caffe AlexNet http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt
caffe GoogleNet http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt
caffe VGG-16 http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt
caffe VGG-19 http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/f43eeefc869d646b449aa6ce66f87bf987a1c9b5/VGG_ILSVRC_19_layers_deploy.prototxt
caffe CaffeNet http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_reference_caffenet/deploy.prototxt
caffe R-CNN http://dl.caffe.berkeleyvision.org/bvlc_reference_rcnn_ilsvrc13.caffemodel https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_reference_rcnn_ilsvrc13/deploy.prototxt

onnx MobileNetV2 https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.onnx
onnx ResNet-18V2 https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v2/resnet18v2.onnx
onnx ResNet-34V2 https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet34v2/resnet34v2.onnx
onnx ResNet-52V2 https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet52v2/resnet52v2.onnx
onnx ResNet-101V2 https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet101v2/resnet101v2.onnx
onnx ResNet-152V2 https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet152v2/resnet152v2.onnx
onnx SqueezeNet https://s3.amazonaws.com/onnx-model-zoo/squeezenet/squeezenet1.1/squeezenet1.1.onnx

# DenseNet is a special case
if [ ! -f "$TEMP_DIR/DenseNet-121.onnx" ];
then
    curl -o densenet121.tar.gz https://s3.amazonaws.com/download.onnx/models/opset_8/densenet121.tar.gz
    tar -xzf densenet121.tar.gz
    mv densenet121/model.onnx $TEMP_DIR/DenseNet-121.onnx
    rm densenet121.tar.gz
    rm -rf densenet121
fi

if [ ! -f "$PROJECT_DIR/models/fp32/DenseNet-121.xml" ];
then
    python3 $INSTALL_DIR/deployment_tools/model_optimizer/mo.py \
        --input_model $TEMP_DIR/DenseNet-121.onnx \
        --output_dir $PROJECT_DIR/models/fp32 \
        --model_name DenseNet-121 \
        --data_type FP32
fi
