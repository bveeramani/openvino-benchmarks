# Copyright (c) 2019 Balaji Veeramani. All rights reserved.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""Functions for benchmarking models using PyTorch as the backend."""
from statistics import median
from timeit import default_timer as timer

import torch
import torchvision.models as models
from tqdm import trange

INPUT_SHAPE = (32, 3, 224, 224)


def main():
    """Main program."""
    print("[Step 1/9] Loading pretrained AlexNet model.")
    alexnet = models.alexnet(pretrained=True)
    print("[Step 2/9] Measuring model performance.")
    latency, throughput = benchmark_sync(alexnet)
    print("[Step 3/9] Dumping statistics.")
    print_results("AlexNet", latency, throughput)

    print("[Step 4/9] Loading pretrained GoogLeNet model.")
    googlenet = models.googlenet(pretrained=True)
    print("[Step 5/9] Measuring model performance.")
    latency, throughput = benchmark_sync(googlenet)
    print("[Step 6/9] Dumping statistics.")
    print_results("GoogLeNet", latency, throughput)

    print("[Step 7/9] Loading pretrained VGG-16 model.")
    vgg16 = models.vgg16(pretrained=True)
    print("[Step 8/9] Measuring model performance.")
    latency, throughput = benchmark_sync(vgg16)
    print("[Step 9/9] Dumping statistics.")
    print_results("VGG-16", latency, throughput)


def benchmark_sync(model, num_trials=3):
    """Measures model performance using synchronous execution.

    Arguments:
        model (nn.Module): A PyTorchm module.
        num_trials (int, optional): The number of trials to run.

    Returns:
        A two-tuple. The first element is the median latency in seconds; the
        second element is the model throughput in frames per second.
    """
    times = []

    for _ in trange(num_trials, leave=False):
        inputs = torch.rand(*INPUT_SHAPE)

        start_time = timer()
        model(inputs)
        end_time = timer()

        time_elapsed = end_time - start_time
        times.append(time_elapsed)

    latency = median(times)
    batch_size = inputs.shape[0]
    throughput = batch_size / latency

    return latency, throughput


def print_results(name, latency, throughput):
    """Formats and prints benchmark results to STDOUT."""
    print("")
    print("Name:\t\t\t%s" % name)
    print("Latency (s):\t\t%.4f" % latency)
    print("Throughput (f/s):\t%.4f" % throughput)
    print("")


if __name__ == "__main__":
    main()
