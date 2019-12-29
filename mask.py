# -*- coding: utf-8 -*-

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# A command line script to classify an image into one of 1000 know objects.
#
# ml mask cvbp [<path>]
#
# Example:
# ml mask cvbp https://cvbp.blob.core.windows.net/public/datasets/object_detection/keypoint_detection.jpg
#
# From the Microsoft Best Practices Suite: Computer Vision
# https://github.com/microsoft/ComputerVision

# ----------------------------------------------------------------------
# Setup.
# ----------------------------------------------------------------------

# Required libraries.

import argparse
import sys
import tempfile
import urllib.request
import utils

from pathlib import Path

from mlhub.pkg import is_url
from mlhub.utils import get_cmd_cwd

from utils_cv.detection.data import coco_labels
from utils_cv.detection.model import DetectionLearner, get_pretrained_maskrcnn
from utils_cv.detection.plot import plot_detections


# ----------------------------------------------------------------------
# Parse command line arguments: path
# ----------------------------------------------------------------------

options = argparse.ArgumentParser(
    prog="mask", description="Detect objects and predict their masks."
)
options.add_argument("path", nargs="+", help="path or url to image")
options.add_argument(
    "-s",
    "--show",
    action="store_true",
    help="display prediction results in image",
)
args = options.parse_args()

# ----------------------------------------------------------------------
# Prediction with pre-trained model
# ----------------------------------------------------------------------

# get pre-trained model
detector = DetectionLearner(
    model=get_pretrained_maskrcnn(), labels=coco_labels()[1:],
)

# predict for
for path in args.path:

    # get image file
    if is_url(path):
        tempdir = tempfile.gettempdir()
        imfile = Path(tempdir) / "temp.jpg"
        urllib.request.urlretrieve(path, imfile)
    else:
        imfile = Path(get_cmd_cwd()) / path

    # predict
    detections = detector.predict(imfile)

    # display results
    if args.show:
        im_res = plot_detections(detections)
        im_res.show()

    # print results
    for box, mask in zip(detections["det_bboxes"], detections["masks"]):
        rle = utils.binary_mask_to_uncompressed_rle(mask)
        height, width = rle["size"]
        counts = ",".join(str(i) for i in rle["counts"])
        sys.stdout.write(
            f"{box.score:.2f},{box.label_name},"
            + f"{box.left},{box.top},{box.right},{box.bottom},"
            + f"{height},{width},{counts},{path}\n"
        )
