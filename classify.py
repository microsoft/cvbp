# -*- coding: utf-8 -*-

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Author: Graham.Williams@microsoft.com
#
# A command line script to classify an image into one of 1000 know objects.
#
# ml classify cvbp [<path>]
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

all_models = [
    "densenet201",
    "alexnet",
    "densenet121",
    "densenet161",
    "densenet169",
    "densenet201",
    "resnet101",
    "resnet152",
    "resnet18",
    "resnet34",
    "resnet50",
    "squeezenet1_0",
    "squeezenet1_1",
    "vgg16_bn",
    "vgg19_bn",
]


# ----------------------------------------------------------------------
# Parse command line arguments: path --model= --webcam=
# ----------------------------------------------------------------------

options = argparse.ArgumentParser(add_help=False)

options.add_argument("path", nargs="*", help="path or url to image")

options.add_argument(
    "-m", "--model", help="model to use (default is resnet18)"
)

options.add_argument(
    "-w", "--webcam", help="which webcam to use (default is 0)"
)

args = options.parse_args()

webcam = 0 if args.webcam is None else args.webcam

if args.model == "list":
    for m in all_models:
        print(m)
    sys.exit(0)
elif args.model is None:
    modeln = ["resnet152"]
elif args.model == "all":
    modeln = all_models
    if not len(args.path):
        sys.stderr.write(
            "Cannot utilise all models from the webcam. "
            + "Do not choose --model=all.\n"
        )
        sys.exit(1)
else:
    modeln = [args.model]

try:
    labels = imagenet_labels()  # The 1000 labels.
except:
    sys.stderr.write(
        "Failed to obtain labels probably because of "
        + "a network connection error.\n"
    )
    sys.exit(1)

# ----------------------------------------------------------------------
# Load the pre-built model
# ----------------------------------------------------------------------

for path in args.path:

    if is_url(path):
        tempdir = tempfile.gettempdir()
        imfile = os.path.join(tempdir, "temp.jpg")
        urllib.request.urlretrieve(path, imfile)
    else:
        imfile = os.path.join(get_cmd_cwd(), path)

    try:
        im = open_image(imfile, convert_mode="RGB")
    except:
        sys.stderr.write(
            f"'{imfile}' may not be an image file and will be skipped.\n"
        )
        continue

    # Select the pre-built model.

    for m in modeln:
        if m == "densenet201":
            model = model_to_learner(
                models.densenet201(pretrained=True), IMAGENET_IM_SIZE
            )
        elif m == "resnet152":
            model = model_to_learner(
                models.resnet152(pretrained=True), IMAGENET_IM_SIZE
            )
        elif m == "alexnet":
            model = model_to_learner(
                models.alexnet(pretrained=True), IMAGENET_IM_SIZE
            )
        elif m == "densenet121":
            model = model_to_learner(
                models.densenet121(pretrained=True), IMAGENET_IM_SIZE
            )
        elif m == "densenet161":
            model = model_to_learner(
                models.densenet161(pretrained=True), IMAGENET_IM_SIZE
            )
        elif m == "densenet169":
            model = model_to_learner(
                models.densenet169(pretrained=True), IMAGENET_IM_SIZE
            )
        elif m == "densenet201":
            model = model_to_learner(
                models.densenet201(pretrained=True), IMAGENET_IM_SIZE
            )
        elif m == "resnet101":
            model = model_to_learner(
                models.resnet101(pretrained=True), IMAGENET_IM_SIZE
            )
        elif m == "resnet152":
            model = model_to_learner(
                models.resnet152(pretrained=True), IMAGENET_IM_SIZE
            )
        elif m == "resnet18":
            model = model_to_learner(
                models.resnet18(pretrained=True), IMAGENET_IM_SIZE
            )
        elif m == "resnet34":
            model = model_to_learner(
                models.resnet34(pretrained=True), IMAGENET_IM_SIZE
            )
        elif m == "resnet50":
            model = model_to_learner(
                models.resnet50(pretrained=True), IMAGENET_IM_SIZE
            )
        elif m == "squeezenet1_0":
            model = model_to_learner(
                models.squeezenet1_0(pretrained=True), IMAGENET_IM_SIZE
            )
        elif m == "squeezenet1_1":
            model = model_to_learner(
                models.squeezenet1_1(pretrained=True), IMAGENET_IM_SIZE
            )
        elif m == "vgg16_bn":
            model = model_to_learner(
                models.vgg16_bn(pretrained=True), IMAGENET_IM_SIZE
            )
        elif m == "vgg19_bn":
            model = model_to_learner(
                models.vgg19_bn(pretrained=True), IMAGENET_IM_SIZE
            )
        else:
            sys.stderr.write(f"Selected model '{m}' is not known.\n")
            sys.exit(1)
        # model = model_to_learner(models.BasicBlock(pretrained=True), IMAGENET_IM_SIZE) # unexpected keyword argument 'pretrained'
        # model = model_to_learner(models.Darknet(pretrained=True), IMAGENET_IM_SIZE) # unexpected keyword argument 'pretrained'
        # model = model_to_learner(models.DynamicUnet(pretrained=True), IMAGENET_IM_SIZE) # missing 2 required positional arguments: 'encoder' and 'n_classes'
        # model = model_to_learner(models.ResLayer(pretrained=True), IMAGENET_IM_SIZE) # unexpected keyword argument 'pretrained'
        # model = model_to_learner(models.ResNet(pretrained=True), IMAGENET_IM_SIZE) # unexpected keyword argument 'pretrained'
        # model = model_to_learner(models.SqueezeNet(pretrained=True), IMAGENET_IM_SIZE) # unexpected keyword argument 'pretrained'
        # model = model_to_learner(models.UnetBlock(pretrained=True), IMAGENET_IM_SIZE) # missing 3 required positional arguments: 'up_in_c', 'x_in_c', and 'hook'
        # model = model_to_learner(models.WideResNet(pretrained=True), IMAGENET_IM_SIZE) # unexpected keyword argument 'pretrained'
        # model = model_to_learner(models.XResNet(pretrained=True), IMAGENET_IM_SIZE) # unexpected keyword argument 'pretrained'
        # model = model_to_learner(models.darknet(pretrained=True), IMAGENET_IM_SIZE) # 'module' object is not callable
        # model = model_to_learner(models.unet(pretrained=True), IMAGENET_IM_SIZE) # 'module' object is not callable
        # model = model_to_learner(models.wrn(pretrained=True), IMAGENET_IM_SIZE) # 'module' object is not callable
        # model = model_to_learner(models.wrn_22(pretrained=True), IMAGENET_IM_SIZE) # got an unexpected keyword argument 'pretrained'
        # model = model_to_learner(models.xception(pretrained=True), IMAGENET_IM_SIZE) # got an unexpected keyword argument 'pretrained'
        # model = model_to_learner(models.xresnet(pretrained=True), IMAGENET_IM_SIZE) #  'module' object is not callable
        # model = model_to_learner(models.xresnet101(pretrained=True), IMAGENET_IM_SIZE) # name 'model_urls' is not defined
        # model = model_to_learner(models.xresnet152(pretrained=True), IMAGENET_IM_SIZE) # name 'model_urls' is not defined
        # model = model_to_learner(models.xresnet18(pretrained=True), IMAGENET_IM_SIZE) # name 'model_urls' is not defined
        # model = model_to_learner(models.xresnet34(pretrained=True), IMAGENET_IM_SIZE) # name 'model_urls' is not defined
        # model = model_to_learner(models.xresnet50(pretrained=True), IMAGENET_IM_SIZE) # name 'model_urls' is not defined

        # Predict the class label.

        _, ind, prob = model.predict(im)
        sys.stdout.write(f"{prob[ind]:.2f},{labels[ind]},{m},{path}\n")

# TODO: Want to load from local copy rather than from ~/.torch which
# means that for a new model the model first needs to be
# downloaded. We might want to cache this download in CONFIGURE.
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

    # Can allow utilise one model for webcam.

    func = partial(classify_frame, model=model, label=labels)

    # ----------------------------------------------------------------------
    # Run webcam to show processed results
    # ----------------------------------------------------------------------

    utils.process_webcam(func, webcam)

    sys.exit(0)
