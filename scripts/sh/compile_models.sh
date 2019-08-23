#!/bin/bash
PROJECT_DIR=$(dirname "$0")/../..
PRECISION=$1

CAFFE_DIR=$PROJECT_DIR/models/caffe
ONNX_DIR=$PROJECT_DIR/models/onnx
INSTALL_DIR=/opt/intel/openvino
MODEL_DIR=$PROJECT_DIR/models/$(echo "$PRECISION" | tr '[:upper:]' '[:lower:]')

if [[ "$PRECISION" != 'FP16' ]] && [[ "$PRECISION" != 'FP32' ]]
then
    echo Expected FP16 or FP32 as an argument.
    exit 9
fi

mkdir -p $CAFFE_DIR
mkdir -p $ONNX_DIR
mkdir -p $MODEL_DIR

caffe()
{
    MODEL_NAME=$1
    CAFFEMODEL_URL=$2
    PROTOTXT_URL=$3
    OUTPUT_LAYER_NAME=$4

    if [ ! -f "$CAFFE_DIR/$MODEL_NAME.caffemodel" ];
    then
        curl -o $CAFFE_DIR/$MODEL_NAME.caffemodel $CAFFEMODEL_URL
    fi

    if [ ! -f "$CAFFE_DIR/$MODEL_NAME.prototxt" ];
    then
        curl -o $CAFFE_DIR/$MODEL_NAME.prototxt $PROTOTXT_URL
    fi

    if [ ! -f "$MODEL_DIR/$MODEL_NAME.xml" ];
    then
        python3 $INSTALL_DIR/deployment_tools/model_optimizer/mo.py \
            --input_model $CAFFE_DIR/$MODEL_NAME.caffemodel \
            --input_proto $CAFFE_DIR/$MODEL_NAME.prototxt \
            --output_dir $MODEL_DIR \
            --model_name $MODEL_NAME \
            --data_type $PRECISION \
            `if [ -z "$OUTPUT_LAYER_NAME" ]; then echo "--output $OUTPUT_LAYER_NAME"; fi`
    fi
}

onnx()
{
    MODEL_NAME=$1
    ONNX_URL=$2
    OUTPUT_LAYER_NAME=$3

    if [ ! -f "$ONNX_DIR/$MODEL_NAME.onnx" ];
    then
        curl -o $ONNX_DIR/$MODEL_NAME.onnx $ONNX_URL
    fi

    if [ ! -f "$MODEL_DIR/$MODEL_NAME.xml" ];
    then
        python3 $INSTALL_DIR/deployment_tools/model_optimizer/mo.py \
            --input_model $ONNX_DIR/$MODEL_NAME.onnx \
            --output_dir $MODEL_DIR \
            --model_name $MODEL_NAME \
            --data_type $PRECISION \
            `if [ -z "$OUTPUT_LAYER_NAME" ]; then echo "--output $OUTPUT_LAYER_NAME"; fi`
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
onnx ResNet-50V2 https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v2/resnet50v2.onnx
onnx ResNet-101V2 https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet101v2/resnet101v2.onnx
onnx ResNet-152V2 https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet152v2/resnet152v2.onnx
onnx SqueezeNet https://s3.amazonaws.com/onnx-model-zoo/squeezenet/squeezenet1.1/squeezenet1.1.onnx

# DenseNet is a special case
if [ ! -f "$ONNX_DIR/DenseNet-121.onnx" ];
then
    curl -o densenet121.tar.gz https://s3.amazonaws.com/download.onnx/models/opset_8/densenet121.tar.gz
    tar -xzf densenet121.tar.gz
    mv densenet121/model.onnx $ONNX_DIR/DenseNet-121.onnx
    rm densenet121.tar.gz
    rm -rf densenet121
fi

if [ ! -f "$MODEL_DIR/DenseNet-121.xml" ];
then
    python3 $INSTALL_DIR/deployment_tools/model_optimizer/mo.py \
        --input_model $ONNX_DIR/DenseNet-121.onnx \
        --output_dir $MODEL_DIR \
        --model_name DenseNet-121 \
        --data_type $PRECISION
fi

caffe AlexNet-conv1 http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt conv1
caffe AlexNet-conv2 http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt conv2
caffe AlexNet-conv3 http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt conv3
caffe AlexNet-conv4 http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt conv4
caffe AlexNet-conv5 http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt conv5
caffe AlexNet-fc6 http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt fc6
caffe AlexNet-fc7 http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt fc7
caffe AlexNet-fc8 http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt fc8

caffe GoogleNet-inception_3b_3x3_reduce http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt inception_3b/3x3_reduce
caffe GoogleNet-inception_4b_3x3_reduce http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt inception_4b/3x3_reduce
caffe GoogleNet-inception_4c_5x5 http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt inception_4c/5x5
caffe GoogleNet-inception_4e_5x5_reduce http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt inception_4e/5x5_reduce
caffe GoogleNet-inception_5b_pool_proj http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt inception_5b/pool_proj
caffe GoogleNet-loss3_classifier http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt loss3/classifier

caffe VGG-16-conv2_1 http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt conv2_1
caffe VGG-16-conv3_1 http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt conv3_1
caffe VGG-16-conv4_1 http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt conv4_1
caffe VGG-16-conv4_3 http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt conv4_3
caffe VGG-16-conv5_3 http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt conv5_3
caffe VGG-16-fc6 http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt fc6
caffe VGG-16-fc7 http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt fc7
caffe VGG-16-fc8 http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt fc8
