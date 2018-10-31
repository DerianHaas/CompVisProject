import matplotlib.image as img
from warp import *
from matchFeatures import *
import os
import cv2

def load(folderName: str, numPics: int, numMatches: int):
    images = []
    Hmatrices = []
    for i in range(numPics-1, 0, -1):
        images.append(cv2.imread("data/"+folderName+"/"+folderName+"-0"+str(i)+".png", 0))
        Hfile = open("data/"+folderName+"/H0"+str(i)+"to0"+str(i-1)+".txt", 'r')
        H = []
        for line in Hfile:
            H.append(list(map(np.float32, line.rstrip('\n').split(' '))))
        Hmatrices.append(H)
    images.append(cv2.imread("data/"+folderName+"/"+folderName+"-00.png", 0))
    images = images[::-1]
    Hmatrices = np.array(Hmatrices[::-1])
    if not os.path.exists("out/"+folderName):
        os.mkdir("out/"+folderName)
    for i in range(len(Hmatrices)-1, -1, -1):
        warpIm, mergeIm = warpImage(images[i+1], images[i], Hmatrices[i])
        img.imsave("out/"+folderName+"/manual"+str(i+1)+"to"+str(i)+"Merge.png", mergeIm, cmap='gray')

        points1, points2, matchIm = matchBruteForce(images[i+1], images[i], numMatches)
        _, mergeBrute = warpImage(images[i+1], images[i], computeH(points1,points2))
        img.imsave("out/" + folderName + "/brute" + str(i + 1) + "to" + str(i) + "Merge.png", mergeIm, cmap='gray')
        img.imsave("out/" + folderName + "/brute" + str(i + 1) + "to" + str(i) + "Matches.png", matchIm, cmap='gray')

        points1, points2, matchIm = matchKnn(images[i + 1], images[i], numMatches)
        _, mergeKnn = warpImage(images[i + 1], images[i], computeH(points1, points2))
        img.imsave("out/" + folderName + "/knn" + str(i + 1) + "to" + str(i) + "Merge.png", mergeIm, cmap='gray')
        img.imsave("out/" + folderName + "/knn" + str(i + 1) + "to" + str(i) + "Matches.png", matchIm, cmap='gray')

        bruteError = computeError2(mergeIm, mergeBrute)
        knnError = computeError2(mergeIm, mergeKnn)
        f = open("out/"+folderName+"/"+str(i + 1) + "to" + str(i) + "Error.txt", "w+")
        f.write("Brute force error (SSIM): "+str(bruteError)+"\n"+"Knn error (SSIM): "+str(knnError))
        f.close()





