from PIL import Image
import numpy as np
import cv2
import pytesseract
import imutils
from pytesseract import Output


def sharpened_image(image: Image) -> Image:
    image = np.array(image) 
    # standardowy kernel do wyostrzania
    kernel = np.array(
        [[0, -1, 0],
        [-1, 5,-1],
        [0, -1, 0]]
    )
    return Image.fromarray(cv2.filter2D(src=image, ddepth=-1, kernel=kernel))


def preprocess_image(image: Image) -> Image:
    #konwersja do skali szarości
    image = image.convert("L")
    open_cv_image = np.array(image) 

    smooth = cv2.GaussianBlur(open_cv_image, (95,95), 0)
    division = cv2.divide(open_cv_image, smooth, scale=255)

    return Image.fromarray(division)