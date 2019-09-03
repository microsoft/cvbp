# -*- coding: utf-8 -*-

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Author: Graham.Williams@microsoft.com
#
# A script for image classfication (optionally with webcam capture)
# based on a model of 1000 known objects.
#
# ml classify cvbp
#
# From the Microsoft Best Practices Suite: Computer Vision
# https://github.com/microsoft/ComputerVision

from fastai.vision import models, Image
from functools import partial
from torchvision.models.detection import fasterrcnn_resnet50_fpn

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
    prog='classify',
    description='Classify objects from camera.'
)

#parser.add_argument(
#    'capture',
#    help="capture live image from webcam")

args = parser.parse_args()

# ----------------------------------------------------------------------
# Prepare processing function
# ----------------------------------------------------------------------

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
#
# If set `os.environ['TORCH_HOME'] = '~/.torch'`, then model weight file would be loaded from '~/.torch/checkpoints/resnet18-5c106cde.pth'.
# See [torch.utils.model_zoo](https://pytorch.org/docs/stable/model_zoo.html#module-torch.utils.model_zoo)
learn = model_to_learner(models.resnet18(pretrained=True), IMAGENET_IM_SIZE)
#learn = model_to_learner(models.resnet152(pretrained=True), IMAGENET_IM_SIZE)
#learn = model_to_learner(models.xresnet152(pretrained=True), IMAGENET_IM_SIZE)

# Want to load from local copy rather than from ~/.torch? Maybe
#learn = load_learner(file="resnet18-5c106cde.pth")

#model = untar_data("resnet18-5c106cde.pth")
#learn = load_learner(model)

func = partial(classify_frame, learner=learn, label=labels)

# ----------------------------------------------------------------------
# Run webcam to show processed results
# ----------------------------------------------------------------------

utils.process_webcam(func)
