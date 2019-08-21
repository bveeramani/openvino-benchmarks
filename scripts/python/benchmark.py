"""Wrapper for benchmark_openvino.py.

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
"""
from benchmark_openvino import main

if __name__ == "__main__":
    main()
