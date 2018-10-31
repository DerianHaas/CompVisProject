import numpy as np
from skimage.measure import compare_ssim


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def computeError(image1: np.ndarray, image2: np.ndarray):
    im1 = rgb2gray(image1)
    im2 = rgb2gray(image2)
    im2.resize(im1.shape)
    im1 = normalize(im1)
    im2 = normalize(im2)
    return np.sum(abs(im1-im2))

def computeError2(image1: np.ndarray, image2: np.ndarray):
    im1 = rgb2gray(image1)
    im2 = rgb2gray(image2)
    im2.resize(im1.shape)
    im1 = normalize(im1)
    im2 = normalize(im2)
    return compare_ssim(im1, im2)

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

def normalize(arr: np.ndarray):
    rng = arr.max()-arr.min()
    amin = arr.min()
    return (arr-amin)*255/rng

def convert(x, y, H):
    point = np.array([x,y,1]).transpose()
    newPoint = H.dot(point)
    newPoint[:2] /= newPoint[2]
    return [newPoint[0], newPoint[1]]