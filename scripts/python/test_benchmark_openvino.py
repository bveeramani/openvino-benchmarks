# Copyright (c) 2019 Balaji Veeramani. All rights reserved.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""Tests for benchmark_openvino.py"""
import os
import random
import string
import unittest

from openvino.inference_engine import IENetwork, IEPlugin

from benchmark_openvino import *

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(os.path.join(ROOT_DIR, "models"), "fp32")

XML_PATH = os.path.join(MODELS_DIR, "squeezenet.xml")
BIN_PATH = os.path.join(MODELS_DIR, "squeezenet.bin")


class BenchmarkTest(unittest.TestCase):

    def test_create_network(self):
        network = create_network(XML_PATH)

        self.assertTrue(isinstance(network, IENetwork))

    def test_set_batch_size(self):
        network = create_network(XML_PATH)

        set_batch_size(network, 32)

        self.assertEqual(network.batch_size, 32)

        input_layer = next(iter(network.inputs))
        shape = network.inputs[input_layer].shape

        self.assertEqual(shape[0], 32)

    def test_get_batch_size(self):
        network = create_network(XML_PATH)
        set_batch_size(network, 32)
        plugin = IEPlugin(device="CPU")
        execution_network = plugin.load(network)

        batch_size = get_batch_size(execution_network)

        self.assertEqual(batch_size, 32)

    def test_create_inputs(self):
        network = create_network(XML_PATH)
        set_batch_size(network, 32)
        plugin = IEPlugin(device="CPU")
        execution_network = plugin.load(network)

        inputs = create_inputs(execution_network)

        self.assertEqual(len(inputs), 1)

        input_layer = next(iter(inputs))
        input_images = inputs[input_layer]

        self.assertEqual(input_images.shape, (32, 3, 224, 224))

    def test_get_inputs(self):
        network = create_network(XML_PATH)
        set_batch_size(network, 32)
        plugin = IEPlugin(device="CPU")
        execution_network = plugin.load(network)

        inputs = get_inputs(execution_network, request_index=0)

        self.assertEqual(len(inputs), 1)

        input_layer = next(iter(inputs))
        input_images = inputs[input_layer]

        self.assertEqual(input_images.shape, (32, 3, 224, 224))

    def test_benchmark_sync(self):
        network = create_network(XML_PATH)
        set_batch_size(network, 32)
        plugin = IEPlugin(device="CPU")
        configure_plugin(plugin, api="sync")
        execution_network = plugin.load(network)

        latency, throughput = benchmark_sync(execution_network, duration=1)

        # I'm assuming that regardless of hardware, the prediction latency will
        # be less than 60s and the throughput will be less than 500 FPS.
        self.assertTrue(0 < latency < 60)
        self.assertTrue(0 < throughput < 500)

    def test_benchmark_async(self):
        network = create_network(XML_PATH)
        set_batch_size(network, 32)
        plugin = IEPlugin(device="CPU")
        configure_plugin(plugin, api="async")
        execution_network = plugin.load(network, num_requests=2)

        latency, throughput = benchmark_async(execution_network, duration=1)

        # I'm assuming that regardless of hardware, the throughput will be less
        # than 500 FPS.
        self.assertEqual(latency, None)
        self.assertTrue(0 < throughput < 500)

    def test_write_results(self):
        results = {
            "name": "alexnet.xml",
            "batch_size": 32,
            "requests": 2,
            "latency": 1,
            "throughput": 1,
            "api": "async",
            "device": "CPU"
        }
        random_filename = "".join([
            random.choice(string.ascii_letters + string.digits)
            for _ in range(32)
        ])

        write_results(results, random_filename)

        with open(random_filename) as file:
            actual = file.read()
        expected = "name,batch_size,requests,latency,throughput,api,device\nalexnet.xml,32,2,1,1,async,CPU\n"

        os.remove(random_filename)
        self.assertEqual(actual, expected)
