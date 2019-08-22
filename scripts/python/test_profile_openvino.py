# Copyright (c) 2019 Balaji Veeramani. All rights reserved.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""Tests for profile_openvino.py"""
import os
import random
import string
import unittest

from profile_openvino import *

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(os.path.join(ROOT_DIR, "models"), "fp32")

XML_PATH = os.path.join(MODELS_DIR, "alexnet.xml")


class ProfileTest(unittest.TestCase):

    def test_get_input_shape(self):
        shape = get_input_shape(XML_PATH)

        self.assertTrue(shape, (3, 227, 227))

    def test_get_precision(self):
        precision = get_precision(XML_PATH)

        self.assertEqual(precision, "FP32")

    def test_measure_size(self):
        size = measure_size(XML_PATH)
        # The size on my device is 243.860896MB.
        self.assertTrue(243.860896 * 0.9 < size < 243.860896 * 1.1)

    def test_count_layers(self):
        num_layers = count_layers(XML_PATH)

        self.assertEqual(num_layers, 22)

    def test_count_convolution_operations(self):
        dom = xml.dom.minidom.parse(XML_PATH)
        layers = dom.getElementsByTagName("layer")

        num_operations = count_convolution_operations(layers[1])

        self.assertEqual(num_operations, 210830400)

    def test_count_fully_connected_operations(self):
        dom = xml.dom.minidom.parse(XML_PATH)
        layers = dom.getElementsByTagName("layer")

        num_operations = count_fully_connected_operations(layers[20])

        self.assertEqual(num_operations, 8192000)

    def test_write_results(self):
        results = {
            "name": "",
            "precision": "",
            "channels": 0,
            "height": 0,
            "width": 0,
            "size": 0,
            "layers": 0,
            "ops": 0
        }
        random_filename = "".join([
            random.choice(string.ascii_letters + string.digits)
            for _ in range(32)
        ])

        write_results(results, random_filename)

        with open(random_filename) as file:
            actual = file.read()
        expected = "name,precision,channels,height,width,size,layers,ops\n,,0,0,0,0,0,0\n"

        os.remove(random_filename)
        self.assertEqual(actual, expected)
