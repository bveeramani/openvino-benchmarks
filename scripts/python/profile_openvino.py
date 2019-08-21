# Copyright (c) 2019 Balaji Veeramani. All rights reserved.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""Tool for profiling models generated with the OpenVINO Model Optimizer.

All operation counts are calculated with respect to a batch size of one.

usage: profile_openvino.py [-h] -m MODEL [-f FILENAME]

Profile Intermediate Representation models.

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model.
  -f FILENAME, --filename FILENAME
                        Optional. Specify the filename where results will be
                        written to.
"""
import argparse
import ast
import os
import xml.dom.minidom

CSV_FIELDS = ("name", "channels", "height", "width", "size", "layers", "ops")


def main():
    """Main program."""
    args = parse_args()

    results = {
        "name": os.path.splitext(os.path.basename(args.model))[0],
        "channels": get_input_shape(args.model)[0],
        "height": get_input_shape(args.model)[1],
        "width": get_input_shape(args.model)[2],
        "size": measure_size(args.model),
        "layers": count_layers(args.model),
        "ops": count_operations(args.model)
    }

    if args.filename:
        write_results(results, args.filename)


def parse_args():
    """Parses CLI arguments and returns a populated Namespace object."""
    parser = argparse.ArgumentParser(
        description="Profile Intermediate Representation models.")
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        required=True,
        help="Required. Path to an .xml file with a trained model.")
    parser.add_argument(
        '-f',
        '--filename',
        type=str,
        required=False,
        default=None,
        help="Optional. Specify the filename where results will be written to.")
    return parser.parse_args()


def get_input_shape(xml_path):
    """Returns the input shape of a model.

    Arguments:
        xml_path (str): Path to an IR XML file.

    Returns:
        A three-tuple containing the number of channels, input height, and
        input width (in that order).
    """
    dom = xml.dom.minidom.parse(xml_path)
    layers = dom.getElementsByTagName("layer")
    for layer in layers:
        if layer.getAttribute("type") == "Input":
            # The Input layer has only one port
            port = layer.getElementsByTagName("port")[0]
            channels = int(
                port.getElementsByTagName("dim")[1].firstChild.nodeValue)
            height = int(
                port.getElementsByTagName("dim")[2].firstChild.nodeValue)
            width = int(
                port.getElementsByTagName("dim")[3].firstChild.nodeValue)
            return channels, height, width

    raise ValueError("model does not have an input layer")


def measure_size(xml_path):
    """Returns the size of the model weights in MB."""
    head, tail = os.path.splitext(xml_path)
    bin_filename = os.path.abspath(head + ".bin")
    return os.path.getsize(bin_filename) / 1e6


def count_layers(xml_path, layer_type=None):
    """Returns the number of layers in a model.

    Arguments:
        xml_path (str): Path to an IR XML file.
        layer_type (str, optional): The type of layer to count.

    Returns:
        The number of layers of the specified type if layer_type is not none;
        otherwise, the total number of layers.
    """
    dom = xml.dom.minidom.parse(xml_path)
    layers = dom.getElementsByTagName("layer")

    if not layer_type:
        return len(layers)

    counter = 0
    for layer in layers:
        if layer.getAttribute("type") == layer_type:
            counter += 1

    return counter


def count_operations(xml_path, layer_type=None):
    """Aproximates the total number of add and multiply operations executed in
    a model while running inference on a single image.

    Only convolution and fully connected layers are counted.

    Arguments:
        xml_path (str): Path to an IR XML file.

    Returns:
        An approximation of the total number of add and multiply operations.
    """
    dom = xml.dom.minidom.parse(xml_path)
    layers = dom.getElementsByTagName("layer")

    num_operations = 0
    for layer in layers:
        if not layer_type or layer.getAttribute("type") == layer_type:
            if layer.getAttribute("type") == "Convolution":
                num_operations += count_convolution_operations(layer)
            elif layer.getAttribute("type") == "FullyConnected":
                num_operations += count_fully_connected_operations(layer)
            else:
                num_operations += 0

    return num_operations


def count_convolution_operations(convolution_layer_element):
    """Returns the number of add and multiply operations executed in a
    convolution layer while running inference on a single image.

    Arguments:
        convolution_layer_element (Element): A layer element with type
            'Convolution'.

    Returns:
        The number of add and multiply operations executed in the layer.

    Raises:
        ValueError: if the value of the type attribute is not 'Convolution'.
    """
    if not convolution_layer_element.getAttribute("type") == "Convolution":
        raise ValueError("expected Convolution layer but got %s." %
                         convolution_layer_element.getAttribute("type"))

    ports = convolution_layer_element.getElementsByTagName("port")

    input_port, output_port = ports[0], ports[1]
    in_channels = int(
        input_port.getElementsByTagName("dim")[1].firstChild.nodeValue)
    out_channels = int(
        output_port.getElementsByTagName("dim")[1].firstChild.nodeValue)

    out_height = int(
        output_port.getElementsByTagName("dim")[2].firstChild.nodeValue)
    out_width = int(
        output_port.getElementsByTagName("dim")[3].firstChild.nodeValue)

    biases = convolution_layer_element.getElementsByTagName("biases")

    layer_data = convolution_layer_element.getElementsByTagName("data")[0]
    groups = int(layer_data.getAttribute("group"))

    kernel_in_channels = in_channels // groups
    kernel_height, kernel_width = ast.literal_eval(
        layer_data.getAttribute("kernel"))
    batch_size = 1  # We calculate ops for a single image

    # ops per output element
    kernel_mul = kernel_height * kernel_width * kernel_in_channels
    kernel_add = kernel_height * kernel_width * kernel_in_channels - 1
    if biases:
        kernel_add += 1
    ops = kernel_mul + kernel_add

    # total ops
    num_out_elements = batch_size * out_channels * out_height * out_width
    total_ops = num_out_elements * ops

    return total_ops


def count_fully_connected_operations(fully_connected_layer_element):
    """Returns the number of add and multiply operations executed in a fully
    connected layer while running inference on a single image.

    Arguments:
        fully_connected_layer_element (Element): A layer element with type
            'FullyConnected'.

    Returns:
        The number of add and multiply operations executed in the layer.

    Raises:
        ValueError: if the value of the type attribute is not 'FullyConnected'.
    """
    if not fully_connected_layer_element.getAttribute(
            "type") == "FullyConnected":
        raise ValueError("expected FullyConnected layer but got %s." %
                         fully_connected_layer_element.getAttribute("type"))

    ports = fully_connected_layer_element.getElementsByTagName("port")

    input_port, output_port = ports[0], ports[1]
    in_features = int(
        input_port.getElementsByTagName("dim")[1].firstChild.nodeValue)
    out_features = int(
        output_port.getElementsByTagName("dim")[1].firstChild.nodeValue)
    biases = fully_connected_layer_element.getElementsByTagName("biases")

    batch_size = 1  # We calculate ops for a single image

    # ops per output element
    total_mul = in_features
    total_add = (in_features - 1)
    if biases:
        total_add += 1

    # total ops
    num_elements = out_features * batch_size
    total_ops = (total_mul + total_add) * num_elements

    return total_ops


def write_results(results, filename):
    """Writes results returned from profiling to a file.

    The data will be serialized in a CSV file. If the output file already
    exists, then the results will be appended to the output file.

    Arguments:
        results (dict): A dictionary containing values for the keys 'name',
        'channels', 'height', 'width', 'size', 'layers', 'ops'.
        filename (str): The path to which the results will be written to.

    Raises:
        ValueError: if results does not contain all of 'name', 'channels',
        'height', 'width', 'size', 'layers', 'ops'.
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


if __name__ == "__main__":
    main()
