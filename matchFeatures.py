import numpy as np
import matplotlib.pyplot as plt
import cv2

def matchBruteForce(img1, img2, numMatches: int, showMatches: bool = False):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    list_kp1 = []
    list_kp2 = []

    # For each match in top 20
    for mat in matches[:numMatches]:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        # Get the coordinates
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        # Append to each list
        list_kp1.append([x1, y1])
        list_kp2.append([x2, y2])


    # Draw first 10 matches.

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:numMatches], None, flags=2)
    if showMatches:
        plt.imshow(img3), plt.show()

    points1 = np.array(list_kp1).transpose()
    points2 = np.array(list_kp2).transpose()
    return points1, points2, img3

def matchKnn(img1, img2, numMatches: int, showMatches: bool = False):
    brisk = cv2.BRISK_create()
    kp1, des1 = brisk.detectAndCompute(img1, None)
    kp2, des2 = brisk.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher()

    # Match descriptors.
    matches = bf.knnMatch(des1, des2, k=2)

    # Sort them in the order of their distance.
    # matches = sorted(matches, key=lambda x: x.distance)

    list_kp1 = []
    list_kp2 = []

    good = [[m] for m,n in matches if m.distance < .75*n.distance]
    good = sorted(good, key=lambda x: x[0].distance)
    # For each match in top 20
    for [mat] in good[:numMatches]:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        # Get the coordinates
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        # Append to each list
        list_kp1.append([x1, y1])
        list_kp2.append([x2, y2])

    # Draw first 10 matches.
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[:numMatches], None, flags=2)
    if showMatches:
        plt.imshow(img3), plt.show()

    points1 = np.array(list_kp1).transpose()
    points2 = np.array(list_kp2).transpose()
    return points1, points2, img3
