# -*- coding: utf-8 -*-

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Author: Graham.Williams@microsoft.com
#
# A script to classify an image into one of 1000 know objects.
#
# ml webcam cvbp
#
# From the Microsoft Best Practices Suite: Computer Vision
# https://github.com/microsoft/ComputerVision

import sys
import os
import urllib.request
import argparse

from mlhub.pkg import is_url
from mlhub.utils import get_cmd_cwd

import fastai
from fastai.vision import models, open_image, load_learner, untar_data

from ipywebrtc import CameraStream, ImageRecorder

# Until this is pip installable we use a local copy!

from utils_cv.common.data import data_path
from utils_cv.common.gpu import which_processor
from utils_cv.classification.data import imagenet_labels
from utils_cv.classification.model import IMAGENET_IM_SIZE, model_to_learner

# ----------------------------------------------------------------------

labels = imagenet_labels()

# models.xresnet50
#learn = model_to_learner(models.resnet18(pretrained=True), IMAGENET_IM_SIZE)
#learn = model_to_learner(models.resnet152(pretrained=True), IMAGENET_IM_SIZE)
#learn = model_to_learner(models.xresnet152(pretrained=True), IMAGENET_IM_SIZE)

# Want to load from local copy rather than from ~/.torch? Maybe
#learn = load_learner(file="resnet18-5c106cde.pth")

#model = untar_data("resnet18-5c106cde.pth")
#learn = load_learner(model)

# Webcam
w_cam = CameraStream(
    constraints={
        'facing_mode': 'user',
        'audio': False,
        'video': { 'width': IMAGENET_IM_SIZE, 'height': IMAGENET_IM_SIZE }
    },
    layout=Layout(width=f'{IMAGENET_IM_SIZE}px')
)
# Image recorder for taking a snapshot
w_imrecorder = ImageRecorder(stream=w_cam, layout=Layout(padding='0 0 0 50px'))
# Label widget to show our classification results
w_label = Label(layout=Layout(padding='0 0 0 50px'))

def classify_frame(_):
    """ Classify an image snapshot by using a pretrained model
    """
    # Once capturing started, remove the capture widget since we don't need it anymore
    if w_imrecorder.layout.display != 'none':
        w_imrecorder.layout.display = 'none'
        
    try:
        im = open_image(io.BytesIO(w_imrecorder.image.value), convert_mode='RGB')
        _, ind, prob = learn.predict(im)
        # Show result label and confidence
        w_label.value = f"{labels[ind]} ({prob[ind]:.2f})"
    except OSError:
        # If im_recorder doesn't have valid image data, skip it. 
        pass
    
    # Taking the next snapshot programmatically
    w_imrecorder.recording = True

# Register classify_frame as a callback. Will be called whenever image.value changes. 
w_imrecorder.image.observe(classify_frame, 'value')

