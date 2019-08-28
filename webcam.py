# -*- coding: utf-8 -*-

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Author: Graham.Williams@microsoft.com
#
# A script for webcam image classfication of 1000 known objects
# or webcam object detection of 90 known objects.
#
# ml webcam cvbp ic
# ml webcam cvbp od
#
# From the Microsoft Best Practices Suite: Computer Vision
# https://github.com/microsoft/ComputerVision

from fastai.vision import models, Image
from functools import partial
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Until this is pip installable we use a local copy!
from utils_cv.classification.data import imagenet_labels
from utils_cv.classification.model import IMAGENET_IM_SIZE, model_to_learner
from utils_cv.detection.data import coco_labels
from utils_cv.detection.model import _get_det_bboxes
from utils_cv.detection.plot import PlotSettings, plot_boxes

import argparse
import utils


# ----------------------------------------------------------------------
# Parse command line arguments
# ----------------------------------------------------------------------

parser = argparse.ArgumentParser(
    prog='webcam',
    description='Classify or Detect objects from camera.'
)

parser.add_argument(
    'domain',
    choices=['ic', 'od'],
    help="image classification (ic)/object detection (od)")

args = parser.parse_args()

print(args.domain)

# ----------------------------------------------------------------------
# Prepare processing function
# ----------------------------------------------------------------------

if args.domain == 'ic':

    # Webcam classification

    def classify_frame(capture, learner, label):
        """Use the learner to predict the class label.
        """
        _, frame = capture.read()  # Capture frame-by-frame
        _, ind, prob = learner.predict(Image(utils.cv2torch(frame)))
        utils.put_text(frame, f"{label[ind]} ({prob[ind]:.2f})")
        return utils.cv2matplotlib(frame)


    labels = imagenet_labels()  # Load model labels

    # Load ResNet model
    # * https://download.pytorch.org/models/resnet18-5c106cde.pth -> ~/.cache/torch/checkpoints/resnet18-5c106cde.pth
    learn = model_to_learner(models.resnet18(pretrained=True), IMAGENET_IM_SIZE)
    #learn = model_to_learner(models.resnet152(pretrained=True), IMAGENET_IM_SIZE)
    #learn = model_to_learner(models.xresnet152(pretrained=True), IMAGENET_IM_SIZE)

    # Want to load from local copy rather than from ~/.torch? Maybe
    #learn = load_learner(file="resnet18-5c106cde.pth")

    #model = untar_data("resnet18-5c106cde.pth")
    #learn = load_learner(model)

    func = partial(classify_frame, learner=learn, label=labels)

elif args.domain == 'od':

    # Webcam object detection

    def detect_frame(capture, model, label):
        """Use the learner to detect objects.
            """
        _, frame = capture.read()  # Capture frame-by-frame
        preds = model([utils.cv2torch(frame)])
        anno_bboxes = _get_det_bboxes(preds, labels=label)
        im_pil = utils.cv2pil(frame)
        plot_boxes(im_pil, anno_bboxes, plot_settings=PlotSettings(rect_color=(0, 255, 0)))
        return utils.pil2matplotlib(im_pil)


    labels = coco_labels()  # Load model labels
    model = fasterrcnn_resnet50_fpn(  # Load ResNet model
        pretrained=True,
        rpn_pre_nms_top_n_test=5,
        rpn_post_nms_top_n_test=5,
        max_size=200,
    )
    model.eval()  # Set model to evaluation mode
    func = partial(detect_frame, model=model, label=labels)

# ----------------------------------------------------------------------
# Run webcam to show processed results
# ----------------------------------------------------------------------

utils.process_webcam(func)
