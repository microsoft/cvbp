# -*- coding: utf-8 -*-

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Author: Graham.Williams@microsoft.com
#
# A command line script to classify an image into one of 1000 know objects.
#
# ml tag cvbp [<path>]
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
import tempfile
import urllib.request

from mlhub.pkg import is_url
from mlhub.utils import get_cmd_cwd

from functools import partial

from fastai.vision import models, open_image, Image

from utils_cv.classification.data import imagenet_labels
from utils_cv.classification.model import IMAGENET_IM_SIZE, model_to_learner

# ----------------------------------------------------------------------
# Parse command line arguments.
# ----------------------------------------------------------------------

options = argparse.ArgumentParser(add_help=False)

options.add_argument(
    'path',
    nargs="*",
    help='path or url to image')

options.add_argument(
    '-m', '--model',
    help="model to use (default is resnet18)")

options.add_argument(
    '-w', '--webcam',
    help="which webcam to use (default is 0)")

args = options.parse_args()

webcam = 0 if args.webcam is None else args.webcam

# ----------------------------------------------------------------------
# Load the ImageNet model - 1000 labels for classification.
# ----------------------------------------------------------------------

try:
    labels = imagenet_labels()   # The 1000 labels.
except:
    sys.stderr.write("Failed to obtain labels probably because of " +
                     "a network connection error.\n")
    sys.exit(1)

# Potential values for the pre-built model: --model=
#
# models.BasicBlock 	models.Darknet 		models.DynamicUnet
# models.ResLayer 	models.ResNet 		models.SqueezeNet
# models.UnetBlock 	models.WideResNet 	models.XResNet
# models.alexnet 	models.darknet 		models.densenet121
# models.densenet161 	models.densenet169 	models.densenet201
# models.resnet101 	models.resnet152 	models.resnet18
# models.resnet34 	models.resnet50 	models.squeezenet1_0
# models.squeezenet1_1 	models.unet 		models.vgg16_bn
# models.vgg19_bn 	models.wrn 		models.wrn_22
# models.xception 	models.xresnet 		models.xresnet101
# models.xresnet152 	models.xresnet18 	models.xresnet34
# models.xresnet50

if args.model == None:
    model = model_to_learner(models.resnet18(pretrained=True), IMAGENET_IM_SIZE)
elif args.model == "resnet152":
    model = model_to_learner(models.resnet152(pretrained=True), IMAGENET_IM_SIZE)
elif args.model == "xresnet152":
    model = model_to_learner(models.xresnet152(pretrained=True), IMAGENET_IM_SIZE)
else:
    sys.stderr.write(f"Selected model '{args.model}' is not known.\n")
    sys.exit(1)

# TODO: Want to load from local copy rather than from ~/.torch which
# means that for a new model the model first needs to be
# downloaded. We might want to cache this downoad in CONFIGURE.
# Some attempts:

# model = load_learner(file="resnet18-5c106cde.pth")
# trdat = untar_data("resnet18-5c106cde.pth")
# model = load_learner(trdat)

# ------------------------------------------------------------------------
# If no args then use the webcam
# ------------------------------------------------------------------------

if not len(args.path):

    # ----------------------------------------------------------------------
    # Prepare processing function
    # ----------------------------------------------------------------------

    def classify_frame(capture, model, label):
        """Use the model to predict the class label.
        """
        _, frame = capture.read()  # Capture frame-by-frame
        _, ind, prob = model.predict(Image(utils.cv2torch(frame)))
        utils.put_text(frame, f"{label[ind]} ({prob[ind]:.2f})")
        return utils.cv2matplotlib(frame)

    func = partial(classify_frame, model=model, label=labels)

    # ----------------------------------------------------------------------
    # Run webcam to show processed results
    # ----------------------------------------------------------------------

    utils.process_webcam(func, webcam)

    sys.exit(0)
    
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
        sys.stderr.write(f"'{imfile}' may not be an image file and will be skipped.\n")
        continue

    # Predict the class label.

    _, ind, prob = model.predict(im)
    sys.stdout.write(f"{prob[ind]:.2f},{labels[ind]},{path}\n")
