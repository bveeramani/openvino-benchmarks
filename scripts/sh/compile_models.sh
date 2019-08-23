#!/bin/bash
PROJECT_DIR=$(dirname "$0")/../..
PRECISION=$1
MODEL_TYPE=$2

CAFFE_DIR=/tmp/caffe
ONNX_DIR=/tmp/onnx
TF_DIR=/tmp/tf
INSTALL_DIR=/opt/intel/openvino
MODEL_DIR=$PROJECT_DIR/models/$(echo "$PRECISION" | tr '[:upper:]' '[:lower:]')

if [[ "$PRECISION" != 'FP16' ]] && [[ "$PRECISION" != 'FP32' ]]
then
    echo Expected FP16 or FP32 as an argument.
    exit 9
fi

mkdir -p $CAFFE_DIR
mkdir -p $ONNX_DIR
mkdir -p $TF_DIR
mkdir -p $MODEL_DIR

caffe()
{
    MODEL_NAME=$1
    CAFFEMODEL_NAME=$2
    CAFFEMODEL_URL=$3
    PROTOTXT_NAME=$4
    PROTOTXT_URL=$5
    OUTPUT_LAYER_NAME=$6

    if [ -f "$MODEL_DIR/$MODEL_NAME.xml" ];
    then
        echo "$MODEL_DIR/$MODEL_NAME has already been compiled."
        return 0
    fi

    if [ ! -f "$CAFFE_DIR/$CAFFEMODEL_NAME" ];
    then
        curl -o $CAFFE_DIR/$CAFFEMODEL_NAME $CAFFEMODEL_URL
    fi

    if [ ! -f "$CAFFE_DIR/$PROTOTXT_NAME" ];
    then
        curl -o $CAFFE_DIR/$PROTOTXT_NAME $PROTOTXT_URL
    fi

    python3 $INSTALL_DIR/deployment_tools/model_optimizer/mo.py \
        --input_model $CAFFE_DIR/$CAFFEMODEL_NAME \
        --input_proto $CAFFE_DIR/$PROTOTXT_NAME \
        --output_dir $MODEL_DIR \
        --model_name $MODEL_NAME \
        --data_type $PRECISION \
        `if [ -z "$OUTPUT_LAYER_NAME" ]; then echo "--output $OUTPUT_LAYER_NAME"; fi`
}

onnx()
{
    MODEL_NAME=$1
    ONNX_NAME=$2
    ONNX_URL=$3
    OUTPUT_LAYER_NAME=$4

    if [ -f "$MODEL_DIR/$MODEL_NAME.xml" ];
    then
        echo "$MODEL_DIR/$MODEL_NAME has already been compiled."
        return 0
    fi

    if [ ! -f "$ONNX_DIR/$ONNX_NAME" ];
    then
        curl -o $ONNX_DIR/$ONNX_NAME $ONNX_URL
    fi

    python3 $INSTALL_DIR/deployment_tools/model_optimizer/mo.py \
        --input_model $ONNX_DIR/$ONNX_NAME \
        --output_dir $MODEL_DIR \
        --model_name $MODEL_NAME \
        --data_type $PRECISION \
        `if [ -z "$OUTPUT_LAYER_NAME" ]; then echo "--output $OUTPUT_LAYER_NAME"; fi`
}

main()
{
    caffe AlexNet \
        AlexNet.caffemodel http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel \
        AlexNet.prototxt https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt

    rm -f $CAFFE_DIR/AlexNet.*

    caffe GoogleNet \
        GoogleNet.caffemodel http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel \
        GoogleNet.prototxt https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt

    rm -f $CAFFE_DIR/GoogleNet.*

    caffe VGG-16 \
        VGG-16.caffemodel http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel \
        VGG-16.prototxt https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt

    rm -f $CAFFE_DIR/VGG-16.*

    caffe VGG-19 \
        VGG-19.caffemodel http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel \
        VGG-16.prototxt https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/f43eeefc869d646b449aa6ce66f87bf987a1c9b5/VGG_ILSVRC_19_layers_deploy.prototxt

    rm -f $CAFFE_DIR/VGG-19.*

    caffe CaffeNet \
        CaffeNet.caffemodel http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel \
        CaffeNet.prototxt https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_reference_caffenet/deploy.prototxt

    rm -f $CAFFE_DIR/CaffeNet.*

    caffe R-CNN \
        R-CNN.caffemodel http://dl.caffe.berkeleyvision.org/bvlc_reference_rcnn_ilsvrc13.caffemodel \
        R-CNN.prototxt https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_reference_rcnn_ilsvrc13/deploy.prototxt

    rm -f $CAFFE_DIR/R-CNN.*

    onnx MobileNetV2 \
        MobileNetV2.onnx https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.onnx

    rm -f $ONNX_DIR/MobileNetV2.*

    onnx ResNet-18V2 \
        ResNet-18V2.onnx https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v2/resnet18v2.onnx

    rm -f $ONNX_DIR/ResNet-18V2.*

    onnx ResNet-34V2 \
        ResNet-34V2.onnx https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet34v2/resnet34v2.onnx

    rm -f $ONNX_DIR/ResNet-34V2.*

    onnx ResNet-50V2 \
        ResNet-50V2.onnx https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v2/resnet50v2.onnx

    rm -f $ONNX_DIR/ResNet-50V2.*

    onnx ResNet-101V2 \
        ResNet-101V2.onnx https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet101v2/resnet101v2.onnx

    rm -f $ONNX_DIR/ResNet-101V2.*

    onnx ResNet-152V2 \
        ResNet-152V2.onnx https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet152v2/resnet152v2.onnx

    rm -f $ONNX_DIR/ResNet-152V2.*

    onnx SqueezeNet \
        SqueezeNet.onnx https://s3.amazonaws.com/onnx-model-zoo/squeezenet/squeezenet1.1/squeezenet1.1.onnx

    rm -f $ONNX_DIR/SqueezeNet.*

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
        rm -f $ONNX_DIR/DenseNet-121.onnx
    fi

    if [ -f "$MODEL_DIR/YOLOV3.xml" ] && [ -f "$MODEL_DIR/YOLOV3-Tiny.xml" ];
    then
        return 0
    fi
    # YOLO-family models
    git clone https://github.com/mystic123/tensorflow-yolo-v3.git $TF_DIR
    git -C $TF_DIR checkout ed60b90
    curl -o $TF_DIR/coco.names https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
    curl -o $TF_DIR/yolov3.weights https://pjreddie.com/media/files/yolov3.weights
    curl -o $TF_DIR/yolov3-tiny.weights https://pjreddie.com/media/files/yolov3-tiny.weights
    python3 $TF_DIR/convert_weights_pb.py --class_names $TF_DIR/coco.names \
        --data_format NHWC --weights_file $TF_DIR/yolov3.weights
    mv frozen_darknet_yolov3_model.pb $TF_DIR/YOLOV3.pb
    python3 $TF_DIR/convert_weights_pb.py --class_names $TF_DIR/coco.names \
        --data_format NHWC --weights_file $TF_DIR/yolov3-tiny.weights --tiny
    mv frozen_darknet_yolov3_model.pb $TF_DIR/YOLOV3-Tiny.pb

    EXTENSIONS_DIR=$INSTALL_DIR/deployment_tools/model_optimizer/extensions

    python3 $INSTALL_DIR/deployment_tools/model_optimizer/mo_tf.py \
        --input_model $TF_DIR/YOLOV3.pb \
        --tensorflow_use_custom_operations_config $EXTENSIONS_DIR/front/tf/yolo_v3.json \
        --output_dir $MODEL_DIR \
        --model_name YOLOV3 \
        --batch 1 \
        --data_type $PRECISION

    python3 $INSTALL_DIR/deployment_tools/model_optimizer/mo_tf.py \
        --input_model $TF_DIR/YOLOV3-Tiny.pb \
        --tensorflow_use_custom_operations_config $EXTENSIONS_DIR/front/tf/yolo_v3_tiny.json \
        --output_dir $MODEL_DIR \
        --model_name YOLOV3-Tiny \
        --batch 1 \
        --data_type $PRECISION

    rm -rf $TF_DIR
}

partial()
{
    caffe AlexNet-conv1 \
        AlexNet.caffemodel http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel \
        AlexNet.prototxt https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt \
        conv1

    caffe AlexNet-conv2 \
        AlexNet.caffemodel http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel \
        AlexNet.prototxt https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt \
        conv2

    caffe AlexNet-conv3 \
        AlexNet.caffemodel http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel \
        AlexNet.prototxt https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt \
        conv3

    caffe AlexNet-conv4 \
        AlexNet.caffemodel http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel \
        AlexNet.prototxt https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt \
        conv4

    caffe AlexNet-conv5 \
        AlexNet.caffemodel http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel \
        AlexNet.prototxt https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt \
        conv5

    caffe AlexNet-fc6 \
        AlexNet.caffemodel http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel \
        AlexNet.prototxt https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt \
        fc6

    caffe AlexNet-fc7 \
        AlexNet.caffemodel http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel \
        AlexNet.prototxt https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt \
        fc7

    caffe AlexNet-fc8 \
        AlexNet.caffemodel http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel \
        AlexNet.prototxt https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt \
        fc8

    rm -f $CAFFE_DIR/AlexNet.*

    caffe GoogleNet-inception_3b_3x3_reduce \
        GoogleNet.caffemodel http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel \
        GoogleNet.prototxt https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt \
        inception_3b/3x3_reduce

    caffe GoogleNet-inception_4b_3x3_reduce \
        GoogleNet.caffemodel http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel \
        GoogleNet.prototxt https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt \
        inception_4b/3x3_reduce

    caffe GoogleNet-inception_4c_5x5 \
        GoogleNet.caffemodel http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel \
        GoogleNet.prototxt https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt \
        inception_4c/5x5

    caffe GoogleNet-inception_4e_5x5_reduce \
        GoogleNet.caffemodel http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel \
        GoogleNet.prototxt https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt \
        inception_4e/5x5_reduce

    caffe GoogleNet-inception_5b_pool_proj \
        GoogleNet.caffemodel http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel \
        GoogleNet.prototxt https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt \
        inception_5b/pool_proj

    caffe GoogleNet-loss3_classifier \
        GoogleNet.caffemodel http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel \
        GoogleNet.prototxt https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt \
        loss3/classifier

    rm -f $CAFFE_DIR/GoogleNet.*

    caffe VGG-16-conv2_1 \
        VGG-16.caffemodel http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel \
        VGG-16.prototxt https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt \
        conv2_1

    caffe VGG-16-conv3_1 \
        VGG-16.caffemodel http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel \
        VGG-16.prototxt https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt \
        conv3_1

    caffe VGG-16-conv4_1 \
        VGG-16.caffemodel http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel \
        VGG-16.prototxt https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt \
        conv4_1

    caffe VGG-16-conv4_3 \
        VGG-16.caffemodel http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel \
        VGG-16.prototxt https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt \
        conv4_3

    caffe VGG-16-conv5_3 \
        VGG-16.caffemodel http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel \
        VGG-16.prototxt https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt \
        conv5_3

    caffe VGG-16-fc6 \
        VGG-16.caffemodel http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel \
        VGG-16.prototxt https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt \
        fc6

    caffe VGG-16-fc7 \
        VGG-16.caffemodel http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel \
        VGG-16.prototxt https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt \
        fc7

    caffe VGG-16-fc8 \
        VGG-16.caffemodel http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel \
        VGG-16.prototxt https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt \
        fc8

    rm -f $CAFFE_DIR/VGG-16.*
}

if [[ "$MODEL_TYPE" == 'PARTIAL' ]];
then
    partial
else
    main
fi

rm -rf $CAFFE_DIR
rm -rf $ONNX_DIR
