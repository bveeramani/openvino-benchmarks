TEMP_DIR=/tmp/models/
INSTALL_DIR=/opt/intel/openvino
MODEL_DIR=~/models

curl -o $TEMP_DIR/AlexNet.caffemodel http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel
curl -o $TEMP_DIR/AlexNet.prototxt https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt
python3 $INSTALL_DIR/deployment_tools/model_optimizer/mo.py \
    --input_model $TEMP_DIR/AlexNet.caffemodel \
    --input_proto $TEMP_DIR/AlexNet.prototxt \
    --output_dir $MODEL_DIR \
    --model_name AlexNet \
    --data_type FP32

# curl -o GoogleNet.caffemodel http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel
# curl -o GoogleNet.prototxt https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt
