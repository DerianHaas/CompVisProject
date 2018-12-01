import numpy as np
import matplotlib.pyplot as plt
import cv2

def detectORB(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    return (kp1, kp2), (des1, des2)

def detectBRISK(img1, img2):
    brisk = cv2.BRISK_create()
    kp1, des1 = brisk.detectAndCompute(img1, None)
    kp2, des2 = brisk.detectAndCompute(img2, None)
    return (kp1, kp2), (des1, des2)

def detectAKAZE(img1, img2):
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)
    return (kp1, kp2), (des1, des2)


def matchBruteForce(img1, img2, keyPoints, descriptors, numMatches: int, showMatches: bool = False):
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # Match descriptors.
    matches = bf.match(descriptors[0], descriptors[1])

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
        (x1, y1) = keyPoints[0][img1_idx].pt
        (x2, y2) = keyPoints[1][img2_idx].pt

        # Append to each list
        list_kp1.append([x1, y1])
        list_kp2.append([x2, y2])


    # Draw first 10 matches.

    img3 = cv2.drawMatches(img1, keyPoints[0], img2, keyPoints[1], matches[:numMatches], None, flags=2)
    if showMatches:
        plt.imshow(img3), plt.show()

    points1 = np.array(list_kp1).transpose()
    points2 = np.array(list_kp2).transpose()
    return points1, points2, img3

def matchBFKnn(img1, img2, keyPoints, descriptors, numMatches: int, showMatches: bool = False):
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # Match descriptors.
    matches = bf.knnMatch(descriptors[0], descriptors[1], 2)

    # Sort them in the order of their distance.
    try:
        good = [[m] for m,n in matches if m.distance < .9*n.distance]
    except ValueError:
        print("Matching bug occured")
        i = 0
        while i < len(matches):
            if len(matches[i]) < 2:
                matches.pop(i)
                i -= 1
            i += 1
        good = [[m] for m, n in matches if m.distance < .9 * n.distance]

    good = sorted(good, key=lambda x: x[0].distance)

    list_kp1 = []
    list_kp2 = []

    # For each match in top 20
    for [mat] in good[:numMatches]:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        # Get the coordinates
        (x1, y1) = keyPoints[0][img1_idx].pt
        (x2, y2) = keyPoints[1][img2_idx].pt

        # Append to each list
        list_kp1.append([x1, y1])
        list_kp2.append([x2, y2])


    # Draw first 10 matches.

    img3 = cv2.drawMatchesKnn(img1, keyPoints[0], img2, keyPoints[1], good[:numMatches], None, flags=2)
    if showMatches:
        plt.imshow(img3), plt.show()

    points1 = np.array(list_kp1).transpose()
    points2 = np.array(list_kp2).transpose()
    return points1, points2, img3



def matchFlannKnn(img1, img2, keyPoints, descriptors, numMatches: int, showMatches: bool = False):
    # create flann object
    flann = cv2.FlannBasedMatcher(dict(algorithm = 6,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1), {})

    # Match descriptors.
    matches = flann.knnMatch(descriptors[0], descriptors[1], k=2)

    # Sort them in the order of their distance.
    # matches = sorted(matches, key=lambda x: x.distance)

    list_kp1 = []
    list_kp2 = []

    try:
        good = [[m] for m,n in matches if m.distance < .75*n.distance]
    except ValueError:
        print("Matching bug occured")
        i = 0
        while i < len(matches):
            if len(matches[i]) < 2:
                matches.pop(i)
                i -= 1
            i += 1
        good = [[m] for m, n in matches if m.distance < .75 * n.distance]

    good = sorted(good, key=lambda x: x[0].distance)
    # For each match in top 20
    for [mat] in good[:numMatches]:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        # Get the coordinates
        (x1, y1) = keyPoints[0][img1_idx].pt
        (x2, y2) = keyPoints[1][img2_idx].pt

        # Append to each list
        list_kp1.append([x1, y1])
        list_kp2.append([x2, y2])

    # Draw first 10 matches.
    img3 = cv2.drawMatchesKnn(img1, keyPoints[0], img2, keyPoints[1], good[:numMatches], None, flags=2)
    if showMatches:
        plt.imshow(img3), plt.show()

    points1 = np.array(list_kp1).transpose()
    points2 = np.array(list_kp2).transpose()
    return points1, points2, img3
