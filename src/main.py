#!/usr/bin/python

import sys, os
import cv2
import numpy as np
from Inpainter import InputState, Inpainter

usageStr = 'Usage: python main.py <pathOfInputImage> <pathOfMaskImage>[ <patchHalfWidth>]'

if __name__ == "__main__":
    if not (3 == len(sys.argv) or 4 == len(sys.argv)):
        print(usageStr)
        exit(-1)
    
    if 3 == len(sys.argv):
        patchHalfWidth = 4
    elif 4 == len(sys.argv):
        try:
            patchHalfWidth = int(sys.argv[3])
        except ValueError:
            print('Unexpected error:', sys.exc_info()[0])
            exit(-1)
    
    # image
    imageFile = sys.argv[1]
    # reads the image with BGR colors
    image = cv2.imread(imageFile, cv2.IMREAD_COLOR)
    if image is None:
        print('Error: Unable to load image file.')
        exit(-1)
    
    # mask
    maskFile = sys.argv[2]
    mask = cv2.imread(maskFile, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print('Error: Unable to load mask file.')
        exit(-1)
    
    i = Inpainter(image, mask, patchHalfWidth)
    if InputState.IS_VALID == i.checkValidInputs():
        i.inpaint()
        cv2.imwrite("../cases/result.jpg", i.result)
        cv2.namedWindow("result")
        cv2.imshow("result", i.image)
        cv2.waitKey()
    else:
        print('Error: invalid parameters.')
