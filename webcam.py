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

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch

from fastai.vision import models, Image
from matplotlib.animation import FuncAnimation

# Until this is pip installable we use a local copy!
from utils_cv.classification.data import imagenet_labels
from utils_cv.classification.model import IMAGENET_IM_SIZE, model_to_learner


# ----------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------

TEXT_COLOR = (0, 255, 0)  # Green
LINE_WIDTH = 2
TEXT_FONT = cv.FONT_HERSHEY_SIMPLEX
TEXT_SIZE = 0.75


def get_camera(num=0):
    """Get the camera.

    :param num:  The camera number.  By default the first camera will be returned.
    :return: The camera used by OpenCV.
    """
    capture = cv.VideoCapture(num)
    if not capture.isOpened():
        print('Unable to load camera!')
        sys.exit(1)

    return capture


def cv2torch(im_cv):
    """Convert image from OpenCV format to PyTorch tensor.

    See https://forums.fast.ai/t/prediction-on-video-input-file/41029/5

    :param im_cv: Image of OpenCV format.
    :return: Image of PyTorch tensor.
    """
    return torch.tensor(np.ascontiguousarray(np.flip(im_cv, 2)).transpose(2, 0, 1)).float() / 255


def put_text(im_cv, text):
    """Put text on the top-middle of image.
    """

    (text_width, text_height), _ = cv.getTextSize(text, TEXT_FONT, TEXT_SIZE, LINE_WIDTH)
    _, img_width, _ = im_cv.shape

    x = int((img_width - text_width) / 2)
    y = text_height + 5
    cv.putText(im_cv, text, (x, y), TEXT_FONT, TEXT_SIZE, TEXT_COLOR, LINE_WIDTH)


def classify_frame(capture, learner, label):
    """Use the learner to predict the class label.
    """
    _, frame = capture.read()  # Capture frame-by-frame
    im = Image(cv2torch(frame))
    _, ind, prob = learner.predict(im)
    text = f"{label[ind]} ({prob[ind]:.2f})"
    put_text(frame, text)
    return cv.cvtColor(frame, cv.COLOR_BGR2RGB)


# ----------------------------------------------------------------------
# Prepare model
# ----------------------------------------------------------------------

# Load model labels
labels = imagenet_labels()

# Load ResNet model
# * https://download.pytorch.org/models/resnet18-5c106cde.pth -> ~/.cache/torch/checkpoints/resnet18-5c106cde.pth
learn = model_to_learner(models.resnet18(pretrained=True), IMAGENET_IM_SIZE)
#learn = model_to_learner(models.resnet152(pretrained=True), IMAGENET_IM_SIZE)
#learn = model_to_learner(models.xresnet152(pretrained=True), IMAGENET_IM_SIZE)

# Want to load from local copy rather than from ~/.torch? Maybe
#learn = load_learner(file="resnet18-5c106cde.pth")

#model = untar_data("resnet18-5c106cde.pth")
#learn = load_learner(model)

# ----------------------------------------------------------------------
# Prepare webcam and plot
# ----------------------------------------------------------------------

# Open webcam
camera = get_camera()

# Turn off axis in plot window
plt.axis('off')

# ----------------------------------------------------------------------
# Image classification on webcam
# ----------------------------------------------------------------------

print("\nPlease close the window (Ctrl-w) to quit.")

im = plt.gca().imshow(classify_frame(camera, learn, labels))
video = FuncAnimation(
    plt.gcf(),
    lambda i: im.set_data(classify_frame(camera, learn, labels)),  # Update plot window with new camera frame
    interval=100)

plt.show()

# When everything is done, release the capture

camera.release()
