#!/usr/bin/python

import sys, os, time
import math, cv2
import numpy as np

class Inpainter():
    DEFAULT_HALF_PATCH_WIDTH=3
    MODE_ADDITION=0
    MODE_MULTIPLICATION=1

    ERROR_INPUT_MAT_INVALID_TYPE=0
    ERROR_INPUT_MASK_INVALID_TYPE=1
    ERROR_MASK_INPUT_SIZE_MISMATCH=2
    ERROR_HALF_PATCH_WIDTH_ZERO=3
    CHECK_VALID=4

    inputImage = None
    mask = updatedMask = None
    result = None
    workImage = None
    sourceRegion = None
    targetRegion = None
    originalSourceRegion = None
    gradientX = None
    gradientY = None
    confidence = None
    data = None
    LAPLACIAN_KERNEL = NORMAL_KERNELX = NORMAL_KERNELY = None
    #cv::Point2i
    bestMatchUpperLeft = bestMatchLowerRight = None
    patchHeight = patchWidth = 0
    #std::vector<cv::Point> -> list[(y,x)]
    fillFront = []
    #std::vector<cv::Point2f> 
    normals = []
    sourcePatchULList = []
    targetPatchSList = []
    targetPatchTList = []
    mode = None
    halfPatchWidth = None
    targetIndex = None

    def __init__(self, inputImage, mask, halfPatchWidth = 4, mode = 1):
        self.inputImage = np.copy(inputImage)
        self.mask = np.copy(mask)
        self.updatedMask = np.copy(mask)
        self.workImage = np.copy(inputImage)
        self.result = np.ndarray(shape = inputImage.shape, dtype = inputImage.dtype)
        self.mode = mode
        self.halfPatchWidth = halfPatchWidth

    def checkValidInputs(self):
        if not self.inputImage.dtype == np.uint8: # TODO: CV_8UC3
            return self.ERROR_INPUT_MAT_INVALID_TYPE
        if not self.mask.dtype == np.uint8: # TODO: CV_8UC1
            return self.ERROR_INPUT_MASK_INVALID_TYPE
        if not self.mask.shape == self.inputImage.shape[:2]: # TODO: CV_ARE_SIZES_EQ
            return self.ERROR_MASK_INPUT_SIZE_MISMATCH
        if self.halfPatchWidth == 0:
            return self.ERROR_HALF_PATCH_WIDTH_ZERO
        return self.CHECK_VALID
    
    def inpaint(self):
#        print 'initialize mats'
        self.initializeMats()
#        print 'calculate gradients'
        self.calculateGradients()
        stay = True
        
        while stay:
#            print 'compute fillFront'
            self.computeFillFront()
#            print 'compute confidence'
            self.computeConfidence()
#            print 'compute data'
            self.computeData()
#            print 'compute target'
            self.computeTarget()
            print 'compute bestpatch', time.asctime()
            self.computeBestPatch()
#            print 'update mats', time.asctime()
            self.updateMats()
            stay = self.checkEnd()

            #cv2.imshow("updatedMask", self.updatedMask)
            cv2.imwrite("../tests/updatedMask.jpg", self.updatedMask)
            #cv2.imshow("inpaint", self.workImage)
            cv2.imwrite("../tests/workImage.jpg", self.workImage)
            #cv2.imshow("gradientX", self.gradientX)
            #cv2.imshow("gradientY", self.gradientY)
            #cv2.waitKey(2);
            print 'One loop Over.'
        
        self.result = np.copy(self.workImage)
        cv2.namedWindow("confidence")
        cv2.imshow("confidence", self.confidence)
    
    def initializeMats(self):
        _, self.confidence = cv2.threshold(self.mask, 10, 255, cv2.THRESH_BINARY)
        _, self.confidence = cv2.threshold(self.confidence, 2, 1, cv2.THRESH_BINARY_INV)
        
        self.sourceRegion = np.copy(self.confidence)
        self.sourceRegion = np.uint8(self.sourceRegion) #dtype = np.uint8
        self.originalSourceRegion = np.copy(self.sourceRegion)
        
        self.confidence = np.float32(self.confidence)
 
#        height, width = self.confidence.shape[:2]
#        for y in range(height):
#            for x in range(width):
#                if not (self.confidence[y, x] == float(1)) and not (self.confidence[y, x] == float(0)):
#                    print y, x, ':', self.confidence[y, x]
#        print 'Checked???'
        _, self.targetRegion = cv2.threshold(self.mask, 10, 255, cv2.THRESH_BINARY)
        _, self.targetRegion = cv2.threshold(self.targetRegion, 2, 1, cv2.THRESH_BINARY)
        self.targetRegion = np.uint8(self.targetRegion)
        self.data = np.ndarray(shape = self.inputImage.shape[:2],  dtype = np.float32)
        
        self.LAPLACIAN_KERNEL = np.ones((3, 3), dtype = np.float32)
        self.LAPLACIAN_KERNEL[1, 1] = -8
        self.NORMAL_KERNELX = np.zeros((3, 3), dtype = np.float32)
        self.NORMAL_KERNELX[1, 0] = -1
        self.NORMAL_KERNELX[1, 2] = 1
        self.NORMAL_KERNELY = cv2.transpose(self.NORMAL_KERNELX)

    def calculateGradients(self):
        srcGray = cv2.cvtColor(self.workImage, cv2.COLOR_RGB2GRAY) # TODO: CV_BGR2GRAY
        
        self.gradientX = cv2.Scharr(srcGray, cv2.CV_32F, 1, 0) # default parameter: scale shoule be 1
        self.gradientX = cv2.convertScaleAbs(self.gradientX)
        self.gradientX = np.float32(self.gradientX)
        self.gradientY = cv2.Scharr(srcGray, cv2.CV_32F, 0, 1)
        self.gradientY = cv2.convertScaleAbs(self.gradientY)
        self.gradientY = np.float32(self.gradientY)
    
#        gx = np.copy(self.workImage)
#        gy = np.copy(self.workImage)
        height, width = self.sourceRegion.shape
        for y in range(height):
            for x in range(width):
                if self.sourceRegion[y, x] == 0:
                    self.gradientX[y, x] = 0
                    self.gradientY[y, x] = 0
#                gx[y, x] = [self.gradientX[y, x]]*3
#                gy[y, x] = [self.gradientY[y, x]]*3
#        
#        cv2.imwrite("../tests/gx.jpg", gx)
#        cv2.imwrite("../tests/gy.jpg", gy)
        self.gradientX /= 255
        self.gradientY /= 255
    
    def computeFillFront(self):
        # elements of boundryMat, whose value > 0 are neighbour pixels of target region. 
        boundryMat = cv2.filter2D(self.targetRegion, cv2.CV_32F, self.LAPLACIAN_KERNEL)
        #print boundryMat.tolist()
        sourceGradientX = cv2.filter2D(self.sourceRegion, cv2.CV_32F, self.NORMAL_KERNELX)
        sourceGradientY = cv2.filter2D(self.sourceRegion, cv2.CV_32F, self.NORMAL_KERNELY)
        #print sourceGradientY.tolist()
        del self.fillFront[:]
        del self.normals[:]
        height, width = boundryMat.shape[:2]
        for y in range(height):
            for x in range(width):
                if boundryMat[y, x] > 0:
                    self.fillFront.append((x, y)) # TODO: reCheck: x before y
                    dx = sourceGradientX[y, x]
                    dy = sourceGradientY[y, x]
                    
                    normalX, normalY = dy, - dx 
                    tempF = math.sqrt(pow(normalX, 2) + pow(normalY, 2))
                    if not tempF == 0:
                        normalX /= tempF
                        normalY /= tempF
                    self.normals.append((normalX, normalY))
    
    def getPatch(self, point):
        centerX, centerY = point
        height, width = self.workImage.shape[:2]
        minX = max(centerX - self.halfPatchWidth, 0)
        maxX = min(centerX + self.halfPatchWidth, width - 1)
        minY = max(centerY - self.halfPatchWidth, 0)
        maxY = min(centerY + self.halfPatchWidth, height - 1)
        upperLeft = (minX, minY)
        lowerRight = (maxX, maxY)
        return upperLeft, lowerRight
    
    def computeConfidence(self):
        for p in self.fillFront:
            pX, pY = p
            (aX, aY), (bX, bY) = self.getPatch(p)
            total = 0
            for y in range(aY, bY + 1):
                for x in range(aX, bX + 1):
                    if self.targetRegion[y, x] == 0:
                        total += self.confidence[y, x]
#                        if not self.confidence[y, x] == float(1):
#                            print y, x, ':', self.confidence[y, x]
            self.confidence[pY, pX] = total / ((bX-aX+1) * (bY-aY+1))
#            print 'total', total
#            print pY, pX, 'confidence set to:', self.confidence[pY, pX]
    
    def computeData(self):
        for i in range(len(self.fillFront)):
            x, y = self.fillFront[i]
            currentNormalX, currentNormalY = self.normals[i]
            self.data[y, x] = math.fabs(self.gradientX[y, x] * currentNormalX + self.gradientY[y, x] * currentNormalY) + 0.001
    
    def computeTarget(self):
        self.targetIndex = 0
        maxPriority, priority = 0, 0
        omega, alpha, beta = 0.7, 0.2, 0.8
        for i in range(len(self.fillFront)):
            x, y = self.fillFront[i]
            # Way 1
            # priority = self.data[y, x] * self.confidence[y, x]
            # Way 2
            Rcp = (1-omega) * self.confidence[y, x] + omega
            priority = alpha * Rcp + beta * self.data[y, x]
            
            if priority > maxPriority:
                maxPriority = priority
                self.targetIndex = i
    
    def computeBestPatch(self):
        minError = bestPatchVariance = 9999999999999999
        currentPoint = self.fillFront[self.targetIndex]
        (aX, aY), (bX, bY) = self.getPatch(currentPoint)
        pHeight, pWidth = bY - aY + 1, bX - aX + 1
        height, width = self.workImage.shape[:2]
        workImage = self.workImage.tolist()
        
        if pHeight != self.patchHeight or pWidth != self.patchWidth:
            # TODO: Gernerate a list, whose elements means that the patch with this element as UpperLeft is usable.
            print 'patch size changed.'
            self.patchHeight, self.patchWidth = pHeight, pWidth
            area = pHeight * pWidth
            SUM_KERNEL = np.ones((pHeight, pWidth), dtype = np.uint8)
            convolvedMat = cv2.filter2D(self.originalSourceRegion, cv2.CV_8U, SUM_KERNEL, anchor = (0, 0))
            self.sourcePatchULList = []
            for y in range(height - pHeight):
                for x in range(width - pWidth):
                    if convolvedMat[y, x] == area:
                        self.sourcePatchULList.append((y, x))
#                        print y, x, 'added.'
    #        print self.originalSourceRegion
    #        print SUM_KERNEL
    #        print convolvedMat.tolist()

        countedNum = 0
        self.targetPatchSList = []
        self.targetPatchTList = []
        for i in range(pHeight):
            for j in range(pWidth):
                if self.sourceRegion[aY+i, aX+j] == 1:
                    countedNum += 1
                    self.targetPatchSList.append((i, j))
                else:
                    self.targetPatchTList.append((i, j))
                    
        
#        for y in range(height - pHeight):
#            for x in range(width - pWidth):
        for (y, x) in self.sourcePatchULList:
                patchError = 0
                meanR = meanG = meanB = 0
                skipPatch = False
                
#                for i in range(pHeight):
#                    for j in range(pWidth):
                for (i, j) in self.targetPatchSList:
#                        if self.originalSourceRegion[y+i, x+j] == 0:
#                            skipPatch = True
#                            print 'really?'
#                            break
#                        if self.sourceRegion[aY+i, aX+j] == 0:
#                            continue
#                        sourcePixel = self.workImage[y+i, x+j]
#                        targetPixel = self.workImage[aY+i, aX+j]
                        sourcePixel = workImage[y+i][x+j]
                        targetPixel = workImage[aY+i][aX+j]
                        
                        for c in range(3):
                            difference = float(sourcePixel[c]) - float(targetPixel[c])
                            patchError += math.pow(difference, 2)
                        meanR += sourcePixel[0]
                        meanG += sourcePixel[1]
                        meanB += sourcePixel[2]
                    
#                    if skipPatch: break
#                if skipPatch: continue
                
                countedNum = float(countedNum)
                patchError /= countedNum
                meanR /= countedNum
                meanG /= countedNum
                meanB /= countedNum
                
                if patchError <= minError:
                    patchVariance = 0
#                    for i in range(pHeight):
#                        for j in range(pWidth):
#                            if self.sourceRegion[aY+i, aX+j] == 0:
                    for (i, j) in self.targetPatchTList:
#                                sourcePixel = self.workImage[y+i, x+j]
                                sourcePixel = workImage[y+i][x+j]
                                difference = sourcePixel[0] - meanR
                                patchVariance += math.pow(difference, 2)
                                difference = sourcePixel[1] - meanG
                                patchVariance += math.pow(difference, 2)
                                difference = sourcePixel[2] - meanB
                                patchVariance += math.pow(difference, 2)
                    
                    if patchError < minError or patchVariance < bestPatchVariance:
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
#        for y in range(pHeight):
#            for x in range(pWidth):
#                if self.sourceRegion[aY+y, aX+x] == 0:
        for (i, j) in self.targetPatchTList:
                    self.workImage[aY+i, aX+j] = self.workImage[bulY+i, bulX+j]
                    self.gradientX[aY+i, aX+j] = self.gradientX[bulY+i, bulX+j]
                    self.gradientY[aY+i, aX+j] = self.gradientY[bulY+i, bulX+j]
                    self.confidence[aY+i, aX+j] = self.confidence[tY, tX]
                    self.sourceRegion[aY+i, aX+j] = 1
                    self.targetRegion[aY+i, aX+j] = 0
                    self.updatedMask[aY+i, aX+j] = 0
    
    def checkEnd(self):
        height, width = self.sourceRegion.shape[:2]
        for y in range(height):
            for x in range(width):
                if self.sourceRegion[y, x] == 0:
                    return True
        return False
    
    
    
    
    
    
    
