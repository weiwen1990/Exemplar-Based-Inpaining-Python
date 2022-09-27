#!/usr/bin/python

import sys, os, time
import math, cv2
import numpy as np
from enum import Enum

class InputState(Enum):
    ERROR_INPUT_MAT_INVALID_TYPE=0
    ERROR_INPUT_MASK_INVALID_TYPE=1
    ERROR_MASK_INPUT_SIZE_MISMATCH=2
    ERROR_PATCH_SIZE_INVALID=3
    IS_VALID=4

class Inpainter():
    image = None
    mask = None

    isTarget = None
    isOriginal = None
    gradientX = None
    gradientY = None
    confidence = None
    data = None

    LAPLACIAN_KERNEL = NORMAL_KERNELX = NORMAL_KERNELY = None
    #cv::Point2i
    bestMatchUpperLeft = bestMatchLowerRight = None
    pHeightLast = pWidthLast = 0
    #std::vector<cv::Point> -> list[(y,x)]
    fillFront = []
    #std::vector<cv::Point2f> 
    normals = []

    sourcePatchULList = []
    targetPatchSList = []
    targetPatchTList = []
    patchHalfWidth = None
    targetIndex = None

    def __init__(self, image, mask, patchHalfWidth):
        self.image = np.copy(image)
        self.mask = np.copy(mask)
        
        self.patchHalfWidth = patchHalfWidth

    def checkValidInputs(self):
        if np.uint8 != self.image.dtype: # CV_8UC3
            return InputState.ERROR_INPUT_MAT_INVALID_TYPE
        if np.uint8 != self.mask.dtype: # CV_8UC1
            return InputState.ERROR_INPUT_MASK_INVALID_TYPE
        if self.mask.shape != self.image.shape[:2]: # CV_ARE_SIZES_EQ
            return InputState.ERROR_MASK_INPUT_SIZE_MISMATCH
        if self.patchHalfWidth <= 0:
            return InputState.ERROR_PATCH_SIZE_INVALID

        return InputState.IS_VALID
    
    def inpaint(self):
        self.initMats()
        self.calcGradients()
        
        while not self.isDone():
            self.computeFillFront()
            self.computeConfidence()
            self.computeData()
            self.computeTarget()
            #print('Computing bestpatch', time.asctime())
            self.computeBestPatch()
            self.updateMats()

            cv2.imwrite("../cases/mask_updating.jpg", self.mask)
            cv2.imwrite("../cases/image_updating.jpg", self.image)
        
        cv2.imshow("Confidence", self.confidence)
    
    def initMats(self):
        _, m = cv2.threshold(self.mask, 10, 255, cv2.THRESH_BINARY)
        _, m = cv2.threshold(m, 2, 1, cv2.THRESH_BINARY)
        self.isTarget = np.uint8(m)
        self.isOriginal = 1 - self.isTarget
        self.confidence = np.float32(self.isOriginal)

        self.data = np.ndarray(shape = self.image.shape[:2], dtype = np.float32)
        
        self.LAPLACIAN_KERNEL = np.ones((3, 3), dtype = np.float32)
        self.LAPLACIAN_KERNEL[1, 1] = -8
        self.NORMAL_KERNELX = np.zeros((3, 3), dtype = np.float32)
        self.NORMAL_KERNELX[1, 0] = -1
        self.NORMAL_KERNELX[1, 2] = 1
        self.NORMAL_KERNELY = cv2.transpose(self.NORMAL_KERNELX)

    def calcGradients(self):
        srcGray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        self.gradientX = cv2.Scharr(srcGray, cv2.CV_32F, 1, 0) # default parameter: scale shoule be 1
        self.gradientX = cv2.convertScaleAbs(self.gradientX)
        self.gradientX = np.float32(self.gradientX)

        self.gradientY = cv2.Scharr(srcGray, cv2.CV_32F, 0, 1)
        self.gradientY = cv2.convertScaleAbs(self.gradientY)
        self.gradientY = np.float32(self.gradientY)
    
        self.gradientX[(self.isTarget == 1)] = 0
        self.gradientY[(self.isTarget == 1)] = 0
        self.gradientX /= 255
        self.gradientY /= 255
    
    def computeFillFront(self):
        # elements of boundryMat, whose value > 0 are neighbour pixels of target region. 
        boundryMat = cv2.filter2D(self.isTarget, cv2.CV_32F, self.LAPLACIAN_KERNEL)

        sourceGradientX = cv2.filter2D(1 - self.isTarget, cv2.CV_32F, self.NORMAL_KERNELX)
        sourceGradientY = cv2.filter2D(1 - self.isTarget, cv2.CV_32F, self.NORMAL_KERNELY)
        
        self.fillFront.clear()
        self.normals.clear()

        h, w = boundryMat.shape[:2]
        for y in range(h):
            for x in range(w):
                if boundryMat[y, x] > 0:
                    self.fillFront.append((x, y))
                    dx = sourceGradientX[y, x]
                    dy = sourceGradientY[y, x]
                    
                    normalX, normalY = dy, -dx 
                    if not (0 == dx and 0 == dy):
                        tempF = math.sqrt(normalX*normalX + normalY*normalY)
                        normalX /= tempF
                        normalY /= tempF
                    self.normals.append((normalX, normalY))
    
    def getPatch(self, point):
        centerX, centerY = point
        h, w = self.image.shape[:2]
        minX = max(centerX - self.patchHalfWidth, 0)
        maxX = min(centerX + self.patchHalfWidth, w - 1)
        minY = max(centerY - self.patchHalfWidth, 0)
        maxY = min(centerY + self.patchHalfWidth, h - 1)
        upperLeft = (minX, minY)
        lowerRight = (maxX, maxY)
        return upperLeft, lowerRight
    
    def computeConfidence(self):
        for p in self.fillFront:
            pX, pY = p
            (aX, aY), (bX, bY) = self.getPatch(p)

            sum = 0
            for y in range(aY, bY + 1):
                for x in range(aX, bX + 1):
                    if self.isTarget[y, x] == 0:
                        sum += self.confidence[y, x]
            self.confidence[pY, pX] = sum / ((bX-aX+1) * (bY-aY+1))
    
    def computeData(self):
        for i in range(len(self.fillFront)):
            x, y = self.fillFront[i]
            currentNormalX, currentNormalY = self.normals[i]
            self.data[y, x] = math.fabs(self.gradientX[y, x] * currentNormalX + self.gradientY[y, x] * currentNormalY) + 1e-5
    
    def computeTarget(self):
        self.targetIndex = 0
        maxPriority, priority = 0, 0
        omega, alpha, beta = 0.7, 0.2, 0.8
        for i in range(len(self.fillFront)):
            x, y = self.fillFront[i]
            # Way 1
            # priority = self.data[y, x] * self.confidence[y, x]
            # Way 2
            rcp = (1-omega) * self.confidence[y, x] + omega
            priority = alpha * rcp + beta * self.data[y, x]
            
            if priority > maxPriority:
                maxPriority = priority
                self.targetIndex = i
    
    def computeBestPatch(self):
        minError = bestPatchVariance = 1e15
        currentPoint = self.fillFront[self.targetIndex]
        (aX, aY), (bX, bY) = self.getPatch(currentPoint)
        pHeight, pWidth = bY - aY + 1, bX - aX + 1
        h, w = self.image.shape[:2]
        
        if pHeight != self.pHeightLast or pWidth != self.pWidthLast:
            self.pHeightLast, self.pWidthLast = pHeight, pWidth
            print('patch size changed to =>', pHeight, pWidth)
            area = pHeight * pWidth
            SUM_KERNEL = np.ones((pHeight, pWidth), dtype = np.uint8)
            convolvedMat = cv2.filter2D(self.isOriginal, cv2.CV_8U, SUM_KERNEL, anchor = (0, 0))
            self.sourcePatchULList = []
            
            # sourcePatchULList: list whose elements is possible to be the UpperLeft of an patch to reference.
            for y in range(h - pHeight):
                for x in range(w - pWidth):
                    if convolvedMat[y, x] == area:
                        self.sourcePatchULList.append((y, x))

        countedNum = 0
        self.targetPatchSList = []
        self.targetPatchTList = []
        
        # targetPatchSList & targetPatchTList: list whose elements are the coordinates of origin/toInpaint pixels.
        for i in range(pHeight):
            for j in range(pWidth):
                if 0 == self.isTarget[aY+i, aX+j]:
                    countedNum += 1
                    self.targetPatchSList.append((i, j))
                else:
                    self.targetPatchTList.append((i, j))
                    
        
        for (y, x) in self.sourcePatchULList:
            patchError = 0
            meanB = meanG = meanR = 0
            skipPatch = False
            
            for (i, j) in self.targetPatchSList:
                sourcePixel = self.image[y+i,x+j]
                targetPixel = self.image[aY+i,aX+j]
                
                for c in range(3):
                    difference = float(sourcePixel[c]) - float(targetPixel[c])
                    patchError += math.pow(difference, 2)
                meanB += sourcePixel[0]
                meanG += sourcePixel[1]
                meanR += sourcePixel[2]
            
            countedNum = float(countedNum)
            patchError /= countedNum
            meanB /= countedNum
            meanG /= countedNum
            meanR /= countedNum
            
            alpha, beta = 0.9, 0.5
            if alpha * patchError <= minError:
                patchVariance = 0
                
                for (i, j) in self.targetPatchTList:
                    sourcePixel = self.image[y+i,x+j]
                    difference = sourcePixel[0] - meanB
                    patchVariance += math.pow(difference, 2)
                    difference = sourcePixel[1] - meanG
                    patchVariance += math.pow(difference, 2)
                    difference = sourcePixel[2] - meanR
                    patchVariance += math.pow(difference, 2)
                
                # Use alpha & Beta to encourage path with less patch variance.
                # For situations in which you need little variance.
                # Alpha = Beta = 1 to disable.
                if patchError < alpha * minError or patchVariance < beta * bestPatchVariance:
                    bestPatchVariance = patchVariance
                    minError = patchError
                    self.bestMatchUpperLeft = (x, y)
                    self.bestMatchLowerRight = (x+pWidth-1, y+pHeight-1)
                    
    
    def updateMats(self):
        targetPoint = self.fillFront[self.targetIndex]
        tX, tY = targetPoint
        (aX, aY), (bX, bY) = self.getPatch(targetPoint)
        bulX, bulY = self.bestMatchUpperLeft
        pHeight, pWidth = bY-aY+1, bX-aX+1
        
        for (i, j) in self.targetPatchTList:
            self.image[aY+i, aX+j] = self.image[bulY+i, bulX+j]
            self.gradientX[aY+i, aX+j] = self.gradientX[bulY+i, bulX+j]
            self.gradientY[aY+i, aX+j] = self.gradientY[bulY+i, bulX+j]
            self.confidence[aY+i, aX+j] = self.confidence[tY, tX]
            self.isTarget[aY+i, aX+j] = 0
            self.mask[aY+i, aX+j] = 0
    
    def isDone(self):
        return (0 == np.count_nonzero(self.isTarget))
