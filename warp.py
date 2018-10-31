from math import *
from compute import *

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


    warpImg = np.zeros((maxRows - minRows, maxCols - minCols), dtype='float')
    if maxRows - minRows > 5000 or maxCols-minCols > 5000:
        return warpImg, warpImg
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

    mergeIm = np.zeros((mergeMaxRows - mergeMinRows, mergeMaxCols - mergeMinCols), dtype='float')

    mergeIm[minRows-mergeMinRows:maxRows-mergeMinRows, minCols-mergeMinCols:maxCols-mergeMinCols] = warpImg[:,:]

    for r in range(0, refIm.shape[0]):
        for c in range(0, refIm.shape[1]):
            mergeIm[r - mergeMinRows, c - mergeMinCols] = refIm[r,c]

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

    warpImg = np.zeros((maxRows - minRows, maxCols - minCols, 3), dtype='float')
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

    mergeIm = np.zeros((mergeMaxRows - mergeMinRows, mergeMaxCols - mergeMinCols, 3), dtype='float')

    mergeIm[-1*mergeMinRows:frameIm.shape[0]-mergeMinRows, -1*mergeMinCols:frameIm.shape[1]-mergeMinCols] = frameIm[:,:]
    mergeIm[minRows - mergeMinRows:maxRows - mergeMinRows, minCols - mergeMinCols:maxCols - mergeMinCols] = warpImg[:,
                                                                                                            :]

    return mergeIm

