# OpenVINO Benchmarks
This repository contains code for reproducing my OpenVINO benchmarking
experiments. You can see performance results [here](https://drive.google.com/open?id=1tNSlOwUzDXjHedvZuMV59DPn-WTDmy6GkywyM1jDacM) and [here](https://docs.google.com/presentation/d/1zRw1FOx5Rbz5KajbIfZz4jI0EFClj1daNtF00OgQVyA/edit#slide=id.g5d444bde14_0_0).

## Benchmark Tool
```
usage: benchmark.py [-h] -m MODEL [-b BATCH_SIZE] [-i NUM_INFER_REQUESTS]
                    [-a {sync,async}] [-d {CPU,GPU,MYRIAD}] [-f FILENAME]

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
  -d {CPU,GPU,MYRIAD}, --device {CPU,GPU,MYRIAD}
                        Optional. Specify a target device to infer on: CPU,
                        GPU, or MYRIAD.
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

## Batch Files

#### Windows
```
usage: vary_api.bat {CPU,MYRIAD}
usage: vary_batch_size.bat {CPU,MYRIAD}
usage: vary_num_infer_requests.bat {CPU,MYRIAD}
```

#### Linux
```
usage: vary_api.sh {CPU,MYRIAD}
usage: vary_batch_size.sh {CPU,MYRIAD}
usage: vary_num_infer_requests.sh {CPU,MYRIAD}
```

## Model Sources
Pre-converted models can be downloaded from [here](https://drive.google.com/drive/folders/1s-K0dAIsJ9OoWfjkasG-wQHd-wpCwja7?usp=sharing).

| Name         | Source                                                                        |
|--------------|-------------------------------------------------------------------------------|
| AlexNet*     | https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet                 |
| GoogleNet    | https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet               |
| DenseNet-121 | https://github.com/onnx/models/tree/master/vision/classification/densenet-121 |
| MobileNetV2  | https://github.com/onnx/models/tree/master/vision/classification/mobilenet    |
| ResNet-18V2  | https://github.com/onnx/models/tree/master/vision/classification/resnet       |
| ResNet-34V2  | https://github.com/onnx/models/tree/master/vision/classification/resnet       |
| ResNet-52V2  | https://github.com/onnx/models/tree/master/vision/classification/resnet       |
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
