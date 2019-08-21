# Copyright (c) 2019 Balaji Veeramani. All rights reserved.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""Wrapper for profile_openvino.py.

All operation counts are calculated with respect to a batch size of one.

usage: profile.py [-h] -m MODEL [-f FILENAME]

Profile Intermediate Representation models.

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model.
  -f FILENAME, --filename FILENAME
                        Optional. Specify the filename where results will be
                        written to.
"""
from profile_openvino import main

if __name__ == "__main__":
    main()
