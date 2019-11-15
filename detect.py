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

from torchvision.models.detection import fasterrcnn_resnet50_fpn

from utils_cv.detection.data import coco_labels
from utils_cv.detection.model import _get_det_bboxes
from utils_cv.detection.plot import plot_boxes, PlotSettings

from fastai.vision import open_image
#from fastai.vision import Image#, models
#from utils_cv.classification.data import imagenet_labels
#from utils_cv.classification.model import IMAGENET_IM_SIZE, model_to_learner

# ----------------------------------------------------------------------
# Parse command line arguments.
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

# Load model labels - length is 91 with a few N/A?

labels = coco_labels()

# Load ResNet model.

model = fasterrcnn_resnet50_fpn(
    pretrained=True,
    rpn_pre_nms_top_n_test=5,
    rpn_post_nms_top_n_test=5,
    max_size=200,
)

# Set model to evaluation mode.
  
model.eval()

if len(args.path):

    for path in args.path:

        if is_url(path):
            tempdir = tempfile.gettempdir()
            imfile = os.path.join(tempdir, "temp.jpg")
            urllib.request.urlretrieve(path, imfile)
        else:
            imfile = os.path.join(get_cmd_cwd(), path)
    
        try:
            im = open_image(imfile, convert_mode='RGB')
        except:
            sys.stderr.write(f"'{imfile}' may not be an image file and" +
                             f"will be skipped.\n")
            continue

        # Output the objects identified.

        preds = model([im.data])
        anno_bboxes = _get_det_bboxes(preds, labels=labels)
        for a in anno_bboxes:
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
        preds = model([utils.cv2torch(frame)])
        anno_bboxes = _get_det_bboxes(preds, labels=label)
        im_pil = utils.cv2pil(frame)
        plot_boxes(im_pil, anno_bboxes,
                   plot_settings=PlotSettings(rect_color=(0, 255, 0)))
        return utils.pil2matplotlib(im_pil)

    func = partial(detect_frame, model=model, label=labels)

    # ----------------------------------------------------------------------
    # Run webcam to show processed results.
    # ----------------------------------------------------------------------

    utils.process_webcam(func, webcam)

    sys.exit(0)


        
