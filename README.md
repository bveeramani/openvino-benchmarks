# OpenVINO Benchmarks
This repository contains code for reproducing my OpenVINO benchmarking
experiments. You can see performance results [here](https://drive.google.com/open?id=1tNSlOwUzDXjHedvZuMV59DPn-WTDmy6GkywyM1jDacM) and [here](https://docs.google.com/presentation/d/1zRw1FOx5Rbz5KajbIfZz4jI0EFClj1daNtF00OgQVyA/edit#slide=id.g5d444bde14_0_0).

## Benchmark Tool
```
usage: benchmark.py [-h] -m MODEL [-b BATCH_SIZE] [-i NUM_INFER_REQUESTS]
                    [-a {sync,async}] [-d {CPU,MYRIAD}] [-f FILENAME]

Benchmark models using OpenVINO.

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model.
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Optional. Specify inference batch size. Default value
                        is 32.
  -i NUM_INFER_REQUESTS, --num_infer_requests NUM_INFER_REQUESTS
                        Optional. Specify number of inference requests.
                        Default value is 2.
  -a {sync,async}, --api {sync,async}
                        Optional. Enable using sync/async API. Default value
                        is sync.
  -d {CPU,MYRIAD}, --device {CPU,MYRIAD}
                        Optional. Specify a target device to infer on: CPU,
                        or MYRIAD.
  -f FILENAME, --filename FILENAME
                        Optional. Specify the filename where data will be
                        written to.
```

## Profile Tool
```
usage: profile.py [-h] -m MODEL [-f FILENAME]

Profile Intermediate Representation models.

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model.
  -f FILENAME, --filename FILENAME
                        Optional. Specify the filename where results will be
                        written to.
```

## Model Compiling tool
```
usage: compile_models.sh <DATA_TYPE {FP16, FP32}> [PARTIAL]

Compile models used in the experiments.

positional arguments:
  DATA_TYPE             Required. Either FP16 or FP32.
  PARTIAL               Optional. Include this argument to compile the partial
                        models instead of the complete models.
```
### Examples
```
$ ./scripts/sh/compile_models.sh FP32
$ ./scripts/sh/compile_models.sh FP16 PARTIAL
```

## Model Sources
Pre-compiled Intermediate Representation models are listed below. Note that the
generated binaries may be incompatible on some systems. You can manually compile
models for your device by using the `scripts/sh/compile_model.sh` tool.

### Complete Models
Pre-compiled models can be downloaded from [here](https://drive.google.com/drive/folders/1s-K0dAIsJ9OoWfjkasG-wQHd-wpCwja7?usp=sharing).

| Name         | Source                                                                        |
|--------------|-------------------------------------------------------------------------------|
| AlexNet*     | https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet                 |
| GoogleNet    | https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet               |
| DenseNet-121 | https://github.com/onnx/models/tree/master/vision/classification/densenet-121 |
| MobileNetV2  | https://github.com/onnx/models/tree/master/vision/classification/mobilenet    |
| ResNet-18V2  | https://github.com/onnx/models/tree/master/vision/classification/resnet       |
| ResNet-34V2  | https://github.com/onnx/models/tree/master/vision/classification/resnet       |
| ResNet-50V2  | https://github.com/onnx/models/tree/master/vision/classification/resnet       |
| ResNet-101V2 | https://github.com/onnx/models/tree/master/vision/classification/resnet       |
| ResNet-152V2 | https://github.com/onnx/models/tree/master/vision/classification/resnet       |
| SqueezeNet   | https://github.com/onnx/models/tree/master/vision/classification/squeezenet   |
| VGG-16       | https://gist.github.com/ksimonyan/211839e770f7b538e2d8                        |
| VGG-19       | https://gist.github.com/ksimonyan/3785162f95cd2d5fee77                        |
| CaffeNet     | https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet      |
| R-CNN        | https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_rcnn_ilsvrc13 |
| YOLOV3       | https://github.com/mystic123/tensorflow-yolo-v3                               |
| YOLOV3-Tiny  | https://github.com/mystic123/tensorflow-yolo-v3                               |

**AlexNet is included as a reference model in the `models/` directory.*

### Partial Models
Additionally, you can download pre-compiled **partial** models from [here](https://drive.google.com/drive/folders/1X7xzMMWTxvVHcnbPB1UisB-ef2k-tJQV?usp=sharing).

| Name                              | Output Layer Name       |
|-----------------------------------|-------------------------|
| AlexNet-conv1                     | conv1                   |
| AlexNet-conv2                     | conv2                   |
| ALexNet-conv3                     | conv3                   |
| AlexNet-conv4                     | conv4                   |
| AlexNet-conv5                     | conv5                   |
| AlexNet-fc6                       | fc6                     |
| AlexNet-fc7                       | fc7                     |
| AlexNet-fc8                       | fc8                     |
| GoogleNet-inception_3b_3x3_reduce | inception_3b/3x3_reduce |
| GoogleNet-inception_4b_3x3_reduce | inception_4b/3x3_reduce |
| GoogleNet-inception_4c_5x5        | inception_4c/5x5        |
| GoogleNet-inception_4e_5x5_reduce | inception_4e/5x5_reduce |
| GoogleNet-inception_5b_pool_proj  | inception_5b/pool_proj  |
| GoogleNet-loss3_classifier        | loss3/classifier        |
| VGG-16-conv2_1                    | conv2_1                 |
| VGG-16-conv3_1                    | conv3_1                 |
| VGG-16-conv4_1                    | conv4_1                 |
| VGG-16-conv4_3                    | conv4_3                 |
| VGG-16-conv5_3                    | conv5_3                 |
| VGG-16-fc6                        | fc6                     |
| VGG-16-fc7                        | fc7                     |
| VGG-16-fc8                        | fc8                     |

### Model Optimizer Environment
* **System Type:** x64-based PC
* **OS:** Windows 10 Pro 10.0.17134 Build 17134
* **Processor:** Intel(R) Core(TM) i7-7660U CPU, 2 Core(s)
* **Disk:** Local SSD
* **Model Optimizer Version:** 2019.1.1-83-g28dfbfd
* **Creation Date:** August 2019

## Known Issues
* A Neural Compute Stick 2 cannot run YOLOV3 with a batch size of 16.
* A Neural Compute Stick 2 cannot run YOLOV3-Tiny with a batch size of 64.

## Demo Output
```
C:\Users\Balaji\Documents\GitHub\openvino-benchmarks>python scripts/python/benchmark.py --device MYRIAD --model models/fp16/AlexNet.xml
[Step 1/7] Constructing plugin for MYRIAD device.
[Step 2/7] Configuring plugin for async execution on Windows.
[Step 3/7] Reading Intermediate Representation of AlexNet.
[Step 4/7] Setting network batch size to 32.
[Step 5/7] Loading network to plugin with 2 inference requests.
[Step 6/7] Measuring performance for 60 seconds.
[Step 7/7] Dumping statistics report.

Name:                   AlexNet
Batch size:             32
Inference requests:     2
Throughput (f/s):       13.9621
API:                    async
Device:                 MYRIAD
```
