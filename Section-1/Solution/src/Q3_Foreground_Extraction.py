# import Library
import skimage
import numpy as np
from matplotlib import pyplot
import cv2

def Q3_solver():
    # import image and getting its parameters
    colorImage = skimage.io.imread("../data/IIScText.png")
    grayImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY)

    # Printing Threshold using inbuilt function
    thresh = int(skimage.filters.threshold_otsu(grayImage))
    print(f"In-built thresh:{thresh}")

    # Generating Binary Image
    binaryImage = grayImage > thresh
    binaryImage = binaryImage * 1
    textIndex = np.where(binaryImage == 1)

    # Background
    imagePath = "../data/IIScMainBuilding.png"
    BGimage = skimage.io.imread(imagePath)
    BGimage[textIndex] = colorImage[textIndex]

    pyplot.title("Otsu's Binarization")
    pyplot.subplot(131)
    pyplot.imshow(colorImage)
    pyplot.subplot(132)
    pyplot.imshow(binaryImage,cmap='gray')
    pyplot.subplot(133)
    pyplot.imshow(BGimage)
    pyplot.show()
