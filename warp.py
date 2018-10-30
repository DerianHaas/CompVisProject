import numpy as np
from math import *

def warpImage(inputIm: np.ndarray, refIm: np.ndarray, H: np.ndarray):

    bounds = [convert(0,0,H),
              convert(0,inputIm.shape[0]-1,H),
              convert(inputIm.shape[1]-1,0,H),
              convert(inputIm.shape[1]-1,inputIm.shape[0]-1,H)]

    bounds = np.array(bounds)

    minCols = int(floor(min(bounds[:,0])))
    minRows = int(floor(min(bounds[:,1])))
    maxCols = int(ceil(max(bounds[:,0])))
    maxRows = int(ceil(max(bounds[:,1])))

    warpImg = np.zeros((maxRows - minRows, maxCols - minCols, 3), dtype='uint8')
    invH = np.linalg.inv(H)
    for r in range(0, maxRows - minRows):
        for c in range(0, maxCols - minCols):
            [x,y] = convert(c + minCols, r + minRows, invH)
            x = int(round(x))
            y = int(round(y))
            if 0 <= x < inputIm.shape[1] and 0 <= y < inputIm.shape[0]:
                warpImg[r, c] = inputIm[y,x]

    mergeMinCols = min([minCols,0])
    mergeMaxCols = max([maxCols, refIm.shape[1]])
    mergeMinRows = min([minRows, 0])
    mergeMaxRows = max([maxRows, refIm.shape[0]])

    mergeIm = np.zeros((mergeMaxRows - mergeMinRows, mergeMaxCols - mergeMinCols, 3), dtype='uint8')

    mergeIm[minRows-mergeMinRows:maxRows-mergeMinRows, minCols-mergeMinCols:maxCols-mergeMinCols] = warpImg[:,:]

    for r in range(0, refIm.shape[0]):
        for c in range(0, refIm.shape[1]):
            mergeIm[r - mergeMinRows, c - mergeMinCols] = (mergeIm[r - mergeMinRows, c - mergeMinCols] * .5 + refIm[r,c] * .25) / .75 if np.sum(mergeIm[r - mergeMinRows, c - mergeMinCols]) else refIm[r,c]

    return warpImg, mergeIm

def frameImage(image, frameIm, H):
    bounds = [convert(0, 0, H),
              convert(0, image.shape[0] - 1, H),
              convert(image.shape[1] - 1, 0, H),
              convert(image.shape[1] - 1, image.shape[0] - 1, H)]

    bounds = np.array(bounds)

    minCols = int(floor(min(bounds[:, 0])))
    minRows = int(floor(min(bounds[:, 1])))
    maxCols = int(ceil(max(bounds[:, 0])))
    maxRows = int(ceil(max(bounds[:, 1])))

    warpImg = np.zeros((maxRows - minRows, maxCols - minCols, 3), dtype='uint8')
    invH = np.linalg.inv(H)
    for r in range(0, maxRows - minRows):
        for c in range(0, maxCols - minCols):
            [x, y] = convert(c + minCols, r + minRows, invH)
            x = int(round(x))
            y = int(round(y))
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                warpImg[r, c] = image[y, x]

    mergeMinCols = min([minCols, 0])
    mergeMaxCols = max([maxCols, frameIm.shape[1]])
    mergeMinRows = min([minRows, 0])
    mergeMaxRows = max([maxRows, frameIm.shape[0]])

    mergeIm = np.zeros((mergeMaxRows - mergeMinRows, mergeMaxCols - mergeMinCols, 3), dtype='uint8')

    mergeIm[-1*mergeMinRows:frameIm.shape[0]-mergeMinRows, -1*mergeMinCols:frameIm.shape[1]-mergeMinCols] = frameIm[:,:]
    mergeIm[minRows - mergeMinRows:maxRows - mergeMinRows, minCols - mergeMinCols:maxCols - mergeMinCols] = warpImg[:,
                                                                                                            :]

    return mergeIm

def computeH(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    numPoints = t1.shape[1]
    ones = np.ones((1, numPoints))
    homo1 = np.append(t1, ones, axis=0).transpose()

    L = np.zeros((2 * numPoints, 9))
    for i in range(0,numPoints):
        L[2 * i, 0:3] = homo1[i]
        L[2 * i, 6:9] = -1 * t2[0, i] * homo1[i]

        L[2 * i + 1, 3:6] = homo1[i]
        L[2 * i + 1, 6:9] = -1 * t2[1, i] * homo1[i]

    h = np.linalg.svd(L)[2][-1,:].reshape((3,3))
    return h


def convert(x, y, H):
    point = np.array([x,y,1]).transpose()
    newPoint = H.dot(point)
    newPoint[:2] /= newPoint[2]
    return [newPoint[0], newPoint[1]]