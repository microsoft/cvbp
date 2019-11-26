# -*- coding: utf-8 -*-

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Author: Graham.Williams@microsoft.com
#
# A command line script to detect objects from 90 known objects.
#
# ml detect cvbp [<path>]
#
# From the Microsoft Best Practices Suite: Computer Vision
# https://github.com/microsoft/ComputerVision

# ----------------------------------------------------------------------
# Setup.
# ----------------------------------------------------------------------

# Required libraries.

import utils

import os
import sys
import argparse

from mlhub.pkg import is_url
from mlhub.utils import get_cmd_cwd

from functools import partial
from PIL import Image

import torchvision

from utils_cv.detection.data import coco_labels
from utils_cv.detection.model import DetectionLearner
from utils_cv.detection.plot import plot_boxes, PlotSettings

# ----------------------------------------------------------------------
# Parse command line arguments: path --model= --webcam=
# ----------------------------------------------------------------------

options = argparse.ArgumentParser(
    prog='detect',
    description='Detect objects from camera.'
)

options.add_argument(
    'path',
    nargs="*",
    help='path or url to image')

options.add_argument(
    '-m', '--model',
    help="model to use (default is resnet50)")

options.add_argument(
    '-w', '--webcam',
    help="which webcam to use (default is 0)")

args = options.parse_args()

webcam = 0 if args.webcam is None else args.webcam

# ----------------------------------------------------------------------
# Prepare processing function
# ----------------------------------------------------------------------

# Load ResNet model.

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    pretrained=True,
    rpn_pre_nms_top_n_test = 5,
    rpn_post_nms_top_n_test = 5,
    max_size=200,
)

detector = DetectionLearner(
    model=model, 
    labels=coco_labels()[1:],  #  First element is '__background__'
)

if len(args.path):

    for path in args.path:

        if is_url(path):
            tempdir = tempfile.gettempdir()
            imfile = os.path.join(tempdir, "temp.jpg")
            urllib.request.urlretrieve(path, imfile)
        else:
            imfile = os.path.join(get_cmd_cwd(), path)
    
        try:
            im = Image.open(imfile).convert('RGB')
        except:
            sys.stderr.write(f"'{imfile}' may not be an image file and " +
                             f"will be skipped.\n")
            continue

        # Output the objects identified.

        detections = detector.predict(im)
        for a in detections['det_bboxes']:
            sys.stdout.write(f"{a.score:.2f},{a.label_name}," +
                             f"{a.left},{a.top},{a.right},{a.bottom}," +
                             f"{path}\n")

else:
    
    # ------------------------------------------------------------------------
    # Webcam object detection
    # ------------------------------------------------------------------------

    def detect_frame(capture, model, label):
        """Use the learner to detect objects.
        """
        _, frame = capture.read()  # Capture frame-by-frame
        #        preds = model([utils.cv2torch(frame)])
        #        anno_bboxes = _get_det_bboxes(preds, labels=label)

        detections = detector.predict(utils.cv2torch(frame))
        im_pil = utils.cv2pil(frame)
        plot_boxes(im_pil, detections['det_bboxes'],
                   plot_settings=PlotSettings(rect_color=(0, 255, 0)))
        return utils.pil2matplotlib(im_pil)

    func = partial(detect_frame, model=model, label=coco_labels()[1:])

    # ----------------------------------------------------------------------
    # Run webcam to show processed results.
    # ----------------------------------------------------------------------

    utils.process_webcam(func, webcam)

    sys.exit(0)


        
