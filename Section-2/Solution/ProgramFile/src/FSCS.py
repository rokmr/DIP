import cv2
import numpy as np
from matplotlib import pyplot
import skimage

# Importing Image
ECE_path = "../data/ECE.png"
IIsc_path = "../data/IIScMain.png"


def import_image(path: str):
    """This function will return Image for given image Path"""
    Image = cv2.imread(path)
    Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
    return Image


def calc_FSCS_img(Image):
    """This function will return fullscale contrast stretched Image"""
    M, N = Image.shape

    # Creating a black Image
    blackStrip = np.zeros([M, 50], dtype='uint8')

    # Getting max & minimum intensity Level
    A = np.amin(Image)
    B = np.amax(Image)

    # Applying FSCS
    I = Image - A
    x = B - A
    ImageFSCS = (I / x) * 255
    ImageFSCS = ImageFSCS.astype(np.uint8)
    return ImageFSCS, blackStrip


def Q1a_solver():

    ECE_Img = import_image(ECE_path)
    ECE_FSCS_Img, b1 = calc_FSCS_img(ECE_Img)

    # Attaching two image
    E = np.concatenate((ECE_Img, b1, ECE_FSCS_Img), axis=1)

    IISc_Img = import_image(IIsc_path)
    IISc_FSCS_Img, b2 = calc_FSCS_img(IISc_Img)

    # Attaching two image
    I = np.concatenate((IISc_Img, b2, IISc_FSCS_Img), axis=1)

    # Plotting Image
    pyplot.figure("ECE")
    pyplot.subplot(121)
    pyplot.title("ECE Original Image")
    pyplot.hist(ECE_Img.ravel(), bins=256, range=(0, 255))
    pyplot.subplot(122)
    pyplot.title("ECE with FSCS")
    pyplot.hist(ECE_FSCS_Img.ravel(), bins=256, range=(0, 255))

    pyplot.figure("IISc")
    pyplot.subplot(121)
    pyplot.title("IISc Original Image")
    pyplot.hist(IISc_Img.ravel(), bins=256, range=(0, 255))
    pyplot.subplot(122)
    pyplot.title("IISc with FSCS")
    pyplot.hist(IISc_FSCS_Img.ravel(), bins=256, range=(0, 255))

    cv2.imshow("ECE", E)
    cv2.imshow("IISc", I)
    pyplot.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
