import matplotlib.pyplot as plt
import matplotlib.image as img
from warp import *
from matchFeatures import *


def displayWarps(im1, im2, points1, points2, filename, frame=False):
    """
    Displays a 2x2 plot of the two images, a warped version of the first image, then the two imaged merged together.
    If frame is true, overwrites the first image with the second when merging.
    Saves the warped and merged images to files using filename.
    """
    H = computeH(points1, points2)
    warpIm, mergeIm = warpImage(im1, im2, H)
    if frame:
        mergeIm = frameImage(im1, im2, H)
    img.imsave("images/warp"+filename, warpIm)
    img.imsave("images/merge"+filename, mergeIm)
    fig = plt.figure(figsize=(2, 2))
    fig.set_size_inches(16.5, 6.5)

    fig.add_subplot(221)
    plt.imshow(im1)


    fig.add_subplot(222)
    plt.imshow(im2)


    fig.add_subplot(223)
    plt.imshow(warpIm)

    fig.add_subplot(224)
    plt.imshow(mergeIm)

    plt.show()

wdc1 = img.imread("images/wdc1.jpg")
wdc2 = img.imread("images/wdc2.jpg")
# points = np.load("images/points.npy")
points1, points2 = matchBruteForce("images/wdc1.jpg", "images/wdc2.jpg", 20, True)
displayWarps(wdc1,wdc2,points1,points2, "Wdc.jpg")
points1, points2 = matchKnn("images/wdc1.jpg", "images/wdc2.jpg", 20, True)
displayWarps(wdc1,wdc2,points1,points2, "Wdc.jpg")