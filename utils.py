import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import PIL
import sys

from matplotlib.animation import FuncAnimation
from torchvision import transforms as T

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


def cv2RGB(im_cv):
    """Convert OpenCV image's color from BGR to RGB.
    """

    return cv.cvtColor(im_cv, cv.COLOR_BGR2RGB)


def cv2matplotlib(im_cv):
    """Convert image from OpenCV format to matplotlib.
    """

    return cv2RGB(im_cv)


def cv2pil(im_cv):
    """Convert image from OpenCV format to PIL.

    See https://stackoverflow.com/a/43234001
    """

    return PIL.Image.fromarray(cv2RGB(im_cv))


def cv2torch(im_cv):
    """Convert image from OpenCV format to PyTorch tensor.

    See https://forums.fast.ai/t/prediction-on-video-input-file/41029/5
    """
    # Equivalent to:
    #     torch.tensor(np.ascontiguousarray(np.flip(im_cv, 2)).transpose(2, 0, 1)).float() / 255
    return T.Compose([T.ToTensor()])(cv2RGB(im_cv))


def pil2matplotlib(im_pil):
    """Convert image from PIL format to matplotlib.
    """
    return np.array(im_pil)


def put_text(im_cv, text):
    """Put text on the top-middle of a OpenCV image.
    """

    (text_width, text_height), _ = cv.getTextSize(text, TEXT_FONT, TEXT_SIZE, LINE_WIDTH)
    _, img_width, _ = im_cv.shape

    x = int((img_width - text_width) / 2)
    y = text_height + 5
    cv.putText(im_cv, text, (x, y), TEXT_FONT, TEXT_SIZE, TEXT_COLOR, LINE_WIDTH)


def process_webcam(func):
    camera = get_camera()  # Open webcam
    plt.axis('off')  # Turn off axis in plot window

    print("\nPlease close the window (Ctrl-w) to quit.")

    im = plt.gca().imshow(func(camera))
    video = FuncAnimation(
        plt.gcf(),
        lambda i: im.set_data(func(camera)),  # Update plot window with new camera frame
        interval=100)

    plt.show()
    camera.release()  # When everything is done, release the capture
