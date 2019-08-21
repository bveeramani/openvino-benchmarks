# Copyright (c) 2019 Balaji Veeramani. All rights reserved.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""Functions for benchmarking models using OpenVINO.

Synchronous benchmarks are measured by repeating inference for a fixed number
of trials (default: 100). Asynchronous benchmarks are measured by repeating
inference as many times as possible in a fixed amount of time (default: 60s).

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
import argparse
import os
import platform
from statistics import median
from timeit import default_timer as timer

import numpy as np
from tqdm import tqdm, trange  # This is used to create progress bars

from openvino.inference_engine import IENetwork, IEPlugin

NUM_SYNC_TRIALS = 100
ASYNC_EXPERIMENT_DURATION = 60

# These are the settings used in the OpenVINO benchmark_app sample
CPU_BIND_THREAD = "YES"
CPU_THROUGHPUT_STREAMS = "2"

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXTENSIONS_DIR = os.path.join(ROOT_DIR, "extensions")

# You need this to run Yolo-family models on CPU
WINDOWS_EXTENSION_PATH = os.path.join(EXTENSIONS_DIR, "cpu_extension.dll")
LINUX_EXTENSION_PATH = os.path.join(EXTENSIONS_DIR, "libcpu_extension_sse4.so")

CSV_FIELDS = ("name", "batch_size", "requests", "latency", "throughput", "api",
              "device")


def main():
    """Main program."""
    args = parse_arguments()
    results = benchmark(xml_path=args.model,
                        batch_size=args.batch_size,
                        num_infer_requests=args.num_infer_requests,
                        api=args.api,
                        device=args.device)

    if args.filename:
        write_results(results, args.filename)


def parse_arguments():
    """Parses CLI arguments and returns a populated Namespace object."""
    parser = argparse.ArgumentParser(
        description="Benchmark models using OpenVINO.")
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        required=True,
        help="Required. Path to an .xml file with a trained model.")
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        required=False,
        default=32,
        help="Optional. Specify inference batch size. Default value is 32.")
    parser.add_argument(
        '-i',
        '--num_infer_requests',
        type=int,
        default=2,
        help=
        "Optional. Specify number of inference requests. Default value is 2.")
    parser.add_argument(
        '-a',
        '--api',
        type=str,
        required=False,
        default='async',
        choices=['sync', 'async'],
        help="Optional. Enable using sync/async API. Default value is sync.")
    parser.add_argument(
        '-d',
        '--device',
        type=str,
        required=False,
        default="CPU",
        choices=["CPU", "GPU", "MYRIAD"],
        help=
        "Optional. Specify a target device to infer on: CPU, GPU, or MYRIAD.")
    parser.add_argument(
        '-f',
        '--filename',
        type=str,
        required=False,
        default=None,
        help="Optional. Specify the filename where data will be written to.")
    return parser.parse_args()


def benchmark(xml_path, batch_size, num_infer_requests, api, device):
    """Benchmarks the specified model using the given settings.

    Arguments:
        xml_path (str): Path to an IR XML file.
        batch_size (int): The desired batch size.
        num_infer_requests (int): The desired number of inference requests.
        api (str): Either 'sync' or 'async'.
        device (str): One of 'CPU', 'GPU', or 'MYRIAD'.

    Returns:
        A dictionary containing values for the keys 'model', 'batch_size',
        'num_infer_requests', 'latency', 'throughput', 'api', and 'device'.

    Raises:
        ValueError: if batch size is not a positive integer.
        ValueError: if num_infer_requests is not a positive integer.
        ValueError: if api is neither 'sync' nor 'async'.
        ValueError: if device is none of 'CPU', 'CPU', and 'MYRIAD'.
    """
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if not isinstance(num_infer_requests, int) or num_infer_requests < 1:
        raise ValueError("num_infer_requests must be a positive integer.")
    if api not in {"sync", "async"}:
        raise ValueError("expected api to be 'sync' or 'async' but got %s." %
                         api)
    if device not in {"CPU", "GPU", "MYRIAD"}:
        raise ValueError(
            "expected api to be 'CPU', 'GPU', or 'MYRIAD' but got %s." % api)

    plugin = IEPlugin(device)
    model_name = os.path.splitext(os.path.basename(xml_path))[0]

    print("[Step 1/6] Configuring plugin for %s execution on %s." %
          (api, platform.system()))
    configure_plugin(plugin, api)

    print("[Step 2/6] Reading Intermediate Representation of %s." % model_name)
    network = create_network(xml_path)

    print("[Step 3/6] Setting network batch size to %d." % batch_size)
    set_batch_size(network, batch_size)

    print("[Step 4/6] Loading network to plugin with %d requests." %
          num_infer_requests)
    execution_network = plugin.load(network, num_infer_requests)

    if api == "sync":
        print("[Step 5/6] Measuring performance over %d trials." %
              NUM_SYNC_TRIALS)
        latency, throughput = benchmark_sync(execution_network,
                                             num_trials=NUM_SYNC_TRIALS)
    elif api == "async":
        print("[Step 5/6] Measuring performance for %d seconds." %
              ASYNC_EXPERIMENT_DURATION)
        latency, throughput = benchmark_async(
            execution_network, duration=ASYNC_EXPERIMENT_DURATION)

    results = {
        "name": model_name,
        "batch_size": batch_size,
        "requests": num_infer_requests,
        "latency": latency,
        "throughput": throughput,
        "api": api,
        "device": device
    }

    print("[Step 6/6] Dumping statistics report.")
    print_results(results)

    del execution_network
    del plugin

    return results


def configure_plugin(plugin, api):
    """Configures a plugin to be run on the target device using the specified api.

    Arguments:
        plugin (IEPlugin): An IEPlugin instance.
        api (str): Either 'sync' or 'async'.

    Raises:
        ValueError: if api is neither 'sync' nor 'async'.
    """
    if api not in {"sync", "async"}:
        raise ValueError("expected api to be 'sync' or 'async' but got %s." %
                         api)

    config = {}

    if plugin.device == "CPU":

        if platform.system() == "Linux":
            plugin.add_cpu_extension(LINUX_EXTENSION_PATH)
        elif platform.system() == "Windows":
            plugin.add_cpu_extension(WINDOWS_EXTENSION_PATH)

        config.update({"CPU_BIND_THREAD": CPU_BIND_THREAD})

        if api == "async":
            config.update({'CPU_THROUGHPUT_STREAMS': CPU_THROUGHPUT_STREAMS})

    plugin.set_config(config)


def create_network(xml_path):
    """Constructs a IENetwork instance using the specified IR XML.

    The .bin file must have the same filename as its associated .xml file.

    Arguments:
        xml_path (str): Path to an IR XML file.

    Returns:
        An IENetwork instance.
    """
    if not os.path.isfile(xml_path):
        raise ValueError("could not find .xml file at the path %s." % xml_path)

    head, tail = os.path.splitext(xml_path)
    bin_path = os.path.abspath(head + ".bin")

    if not os.path.isfile(bin_path):
        raise ValueError("could not find .bin file at the path %s." % bin_path)

    network = IENetwork(model=xml_path, weights=bin_path)
    return network


def set_batch_size(network, batch_size):
    """Sets the batch size of the given network.

    Note that this function will NOT work with an ExecutableNetwork instance.

    Arguments:
        network (IENetwork): An IENetwork instance.

    Raises:
        NotImplementedError: if network has more than one input layer.
        ValueError: if batch size is not a positive integer.
    """
    if len(network.inputs) != 1:
        raise NotImplementedError("expected one input layer but got %d." %
                                  len(network.inputs))
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")

    input_layer = next(iter(network.inputs))
    shape = network.inputs[input_layer].shape

    shape[0] = batch_size

    network.batch_size = batch_size
    network.reshape({input_layer: shape})


def get_batch_size(execution_network):
    """Returns the batch size of the given executable network.

    Note that this function will NOT work with an IENetwork instance.

    Arguments:
        execution_network (ExecutableNetwork): An ExecutableNetwork instance.

    Returns:
        The batch size of the network.

    Raises:
        NotImplementedError: if network has more than one input layer.
    """
    assert execution_network.requests
    inputs = execution_network.requests[0].inputs

    if len(inputs) != 1:
        raise NotImplementedError("expected one input layer but got %d." %
                                  len(inputs))
    input_layer = next(iter(inputs))
    shape = inputs[input_layer].shape

    return shape[0]


def create_inputs(execution_network):
    """Randomly generates inputs for an execution network.

    Arguments:
        execution_network (ExecutableNetwork): An ExecutableNetwork instance.

    Returns:
        A dictionary that maps input layers to numpy.ndarray objects of proper
        shape.

    Raises:
        NotImplementedError: if execution_network has more than one input layer.
    """
    assert execution_network.requests
    inputs = execution_network.requests[0].inputs

    if len(inputs) != 1:
        raise NotImplementedError("expected one input layer but got %d." %
                                  len(inputs))

    input_layer = next(iter(inputs))
    shape = inputs[input_layer].shape
    input_images = np.random.rand(*shape)

    return {input_layer: input_images}


def benchmark_sync(execution_network, num_trials):
    """Benchmark the given execution network using synchronous inference.

    Arguments:
        execution_network (ExecutableNetwork): An ExecutableNetwork instance.
        num_trials (int): The number of inference trials to run.

    Returns:
        A two-tuple. The first element is the median prediction latency, and the
        second element is the model throughput.

    Raises:
        ValueError: if num_trials is not a positive integer
    """
    if not isinstance(num_trials, int) or num_trials < 1:
        raise ValueError("num_trials must be a positive integer.")

    inference_inputs = create_inputs(execution_network)
    execution_network.infer(inference_inputs)  # cache warmup

    times = []
    for _ in trange(num_trials, leave=False):
        inference_inputs = create_inputs(execution_network)

        start_time = timer()
        execution_network.infer(inference_inputs)
        end_time = timer()

        time_elapsed = end_time - start_time
        times.append(time_elapsed)

    times.sort()

    latency = median(times)
    batch_size = get_batch_size(execution_network)
    throughput = batch_size / latency

    return latency, throughput


def benchmark_async(execution_network, duration):
    """Benchmark the given execution network using asynchronous inference.

    The median prediction latency is undefined for asynchronous inference. So,
    the returned prediction latency will always be None.

    benchmark_sync generates new inputs every iterations, but benchmark_async
    uses the same inputs for the entire duration. This behaviour can also be
    found in the OpenVINO benchmark_app sample, on which this function is based.

    See https://github.com/opencv/dldt/blob/2019/inference-engine/ie_bridges/
        python/sample/benchmark_app/benchmark/benchmark.py.
    This function was checked with commit 693ab4e.

    Arguments:
        execution_network (ExecutableNetwork): An ExecutableNetwork instance.
        duration (float): A non-negative number.

    Returns:
        A two-tuple. The first element is the median prediction latency, and the
        second element is the model throughput.
    """
    if duration < 0:
        raise ValueError("duration must be a non-negative number.")

    inference_inputs = create_inputs(execution_network)
    inference_requests = execution_network.requests

    required_inference_requests_were_executed = False

    current_inference = 0
    previous_inference = 1 - len(inference_requests)
    iterations_completed = 0

    # warming up - out of scope
    inference_requests[0].async_infer(inference_inputs)
    inference_requests[0].wait()

    progress_bar = tqdm(total=duration,
                        leave=False,
                        bar_format="{l_bar}{bar}| {elapsed}")

    time_elapsed = 0
    while not required_inference_requests_were_executed or time_elapsed < duration:
        iteration_start_time = timer()

        execution_network.start_async(current_inference, inference_inputs)

        if previous_inference >= 0:
            inference_requests[previous_inference].wait()

        current_inference += 1
        if current_inference >= len(inference_requests):
            current_inference = 0
            required_inference_requests_were_executed = True

        previous_inference += 1
        if previous_inference >= len(inference_requests):
            previous_inference = 0

        iterations_completed += 1

        iteration_duration = timer() - iteration_start_time
        time_elapsed += iteration_duration

        progress_bar.update(iteration_duration)

    progress_bar.close()

    # wait the latest inference executions
    for incomplete_request in inference_requests:
        if incomplete_request.wait(0) != 0:
            incomplete_request.wait()

    batch_size = get_batch_size(execution_network)
    throughput = batch_size * iterations_completed / time_elapsed

    return None, throughput


def write_results(results, filename):
    """Writes results returned from benchmark to a file.

    The data will be serialized in a CSV file. If the output file already
    exists, then the results will be appended to the output file.

    Arguments:
        results (dict): A dictionary containing values for the keys 'name',
            'batch_size', 'requests', 'latency', 'throughput', 'api', and
            'device'.
        filename (str): The path to which the results will be written to.

    Raises:
        ValueError: if results does not contain all of 'name', 'batch_size',
            'requests', 'latency', 'throughput', 'api', or 'device'.
    """
    for field in CSV_FIELDS:
        if field not in results:
            raise ValueError(
                "expected results to contain the key %s, but the key could not be found."
                % field)

    need_heading = not os.path.isfile(filename) or not os.path.getsize(filename)

    with open(filename, "a") as file:
        if need_heading:
            heading = ",".join(CSV_FIELDS)
            file.write(heading + "\n")

        values = [str(results[field]) for field in CSV_FIELDS]
        row = ",".join(values)
        file.write(row + "\n")


def print_results(results):
    """Writes results returned from benchmark to STDOUT.

    Arguments:
        results (dict): A dictionary containing values for the keys 'name',
            'batch_size', 'requests', 'latency', 'throughput', 'api', and
            'device'.

    Raises:
        ValueError: if results does not contain all of 'name', 'batch_size',
            'requests', 'latency', 'throughput', 'api', or 'device'.
    """
    for field in CSV_FIELDS:
        if field not in results:
            raise ValueError(
                "expected results to contain the key %s, but the key could not be found."
                % field)

    print("")
    print("Name:\t\t\t%s" % results["name"])
    print("Batch size:\t\t%s" % results["batch_size"])
    print("Inference requests:\t%s" % results["requests"])
    if results["latency"]:
        print("Latency (s):\t\t%s" % results["latency"])
    print("Throughput (f/s):\t%s" % results["throughput"])
    print("API:\t\t\t%s" % results["api"])
    print("Device:\t\t\t%s" % results["device"])


if __name__ == "__main__":
    main()
