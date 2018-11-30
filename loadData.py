import matplotlib.image as img
from warp import *
from matchFeatures import *
import os
import cv2

detectAlgs = {'ORB': detectORB, 'BRISK': detectBRISK, 'AKAZE': detectAKAZE}
matchAlgs = {'brute': matchBruteForce, 'bruteKnn': matchBFKnn, 'flann': matchFlannKnn}


def load(folderName: str, numPics: int, numMatches: int):
    images = []
    Hmatrices = []
    for i in range(numPics - 1, 0, -1):
        num = ("0" if i < 10 else "") + str(i)
        num2 = ("0" if i < 11 else "") + str(i - 1)
        images.append(cv2.imread("data/" + folderName + "/" + folderName + "-" + num + ".png", 0))
        Hfile = open("data/" + folderName + "/H" + num + "to" + num2 + ".txt", 'r')
        H = []
        for line in Hfile:
            H.append(list(map(np.float32, line.rstrip('\n').split(' '))))
        Hmatrices.append(H)
    images.append(cv2.imread("data/" + folderName + "/" + folderName + "-00.png", 0))
    images = images[::-1]
    Hmatrices = np.array(Hmatrices[::-1])
    if not os.path.exists("out/" + folderName):
        os.mkdir("out/" + folderName)
    if not os.path.exists("out/" + folderName + "/Error"):
        os.mkdir("out/" + folderName + "/Error")
    if not os.path.exists("out/" + folderName + "/Manual"):
        os.mkdir("out/" + folderName + "/Manual")
    for detect in detectAlgs:
        if not os.path.exists("out/" + folderName + "/" + detect):
            os.mkdir("out/" + folderName + "/" + detect)
        for match in matchAlgs:
            if not os.path.exists("out/" + folderName +  "/" + detect + "/" + match):
                os.mkdir("out/" + folderName +  "/" + detect + "/" + match)

    for i in range(len(Hmatrices) - 1, -1, -1):
        num = ("0" if i < 10 else "") + str(i)
        num2 = ("0" if i < 9 else "") + str(i + 1)

        print("Matching images "+str(num2)+" to "+str(num)+" in dataset "+folderName+" with given homography")

        _, mergeImManual = warpImage(images[i + 1], images[i], Hmatrices[i])
        img.imsave("out/" + folderName + "/Manual/" + num2 + "to" + num + "Merge.png", mergeImManual, cmap='gray')

        f = open("out/" + folderName + "/Error/" + num2 + "to" + num + ".txt", "w+")

        for detect in detectAlgs:
            keyPoints, descriptors = detectAlgs[detect](images[i + 1], images[i])
            f.write("Detection Algorithm: " + detect + "\n")
            for match in matchAlgs:
                print("Matching images "+str(num2)+" to "+str(num)+" in dataset "+folderName+" with "+detect+" detection and "+match+" matching")
                points1, points2, matchIm = matchAlgs[match](images[i + 1], images[i], keyPoints, descriptors, numMatches)
                _, mergeIm = warpImage(images[i + 1], images[i], computeH(points1, points2))
                if type(mergeIm) != str:
                    img.imsave("out/" + folderName +  "/" + detect + "/" + match + "/" + num2 + "to" + num + "Merge.png", mergeIm, cmap='gray')
                    img.imsave("out/" + folderName +  "/" + detect + "/" + match + "/" + num2 + "to" + num + "Matches.png", matchIm, cmap='gray')
                    error = computeError2(mergeImManual, mergeIm)
                    f.write("\tMatching Algorithm - " + match + " - Error (SSIM): " + str(error) + "\n")
                else:
                    f.write("\tMatching Algorithm - " + match + " - Error Occured during Warp!")
                    print('Warping bug occured')
        f.close()
