import cv2
import HE
import FSCS
import numpy as np


def clahe_clip_cdf_distribute(Img, clipLimit):
    """This function will take image and clipping limit and will distribute the intensity grain and return cdf of
    clipped histogram """

    # Geting CDF and Pdf of image
    pdf_img, cdf_img = HE.pdf_cdf_calculator(Img)

    # Getting Intensity which are above and below clip Limit
    intensityAbove_cliplimit = (pdf_img > clipLimit) * 1
    intensityBelow_cliplimit = (pdf_img <= clipLimit) * 1

    # Calculation for intensity above clip limit
    pdfAbove_clipLimit = (pdf_img - clipLimit) * intensityAbove_cliplimit
    total_pdfAbove_cliplimt = sum(pdfAbove_clipLimit)

    # Distibution on intensity
    s=int(sum(intensityBelow_cliplimit))

    if s!=0:
        average_intensity_to_be_distributed = total_pdfAbove_cliplimt / s
    clahe_clip_pdf = (pdf_img + average_intensity_to_be_distributed) * intensityBelow_cliplimit + intensityAbove_cliplimit * clipLimit

    # Calculation of CDF
    cdf = np.zeros(shape=(256,))
    cdf[0] = clahe_clip_pdf[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + clahe_clip_pdf[i]

    return cdf


def clahe_without_overlap(path, name, clip):
    """ Function to get clahe without overlap Image provided path of image, name to be put and clip"""
    # Importing Image
    Image = FSCS.import_image(path)
    M, N = Image.shape

    # Blank Image
    blankImage = np.zeros([M, N], dtype=int)

    # Forming Patches
    M_inc = int(M / 8)
    N_inc = int(N / 8)

    for i in range(0, M, M_inc):
        for j in range(0, N, N_inc):
            temp = Image[i:i + M_inc, j:j + N_inc]
            c = clahe_clip_cdf_distribute(temp, clip)
            Clahe_patch_Img = HE.hist_eq(temp, c)
            blankImage[i:i + M_inc, j:j + N_inc] = Clahe_patch_Img

    blankImage = blankImage.astype(np.uint8)
    cv2.imshow(name, blankImage)


def clahe_with_overlap(path: str, name: str, overLapPercent: int, clip):
    """ Function to get clahe with overlap Image provided path of image, name to be put, overlap percentage and clip"""
    Image = FSCS.import_image(path)
    M, N = Image.shape
    x = 1 + (overLapPercent / 100)

    blankImage = np.zeros([M, N], dtype=int)

    # Forming Patches
    h = int(M / 8)
    w = int(N / 8)

    # Making overlap by decreasing then next patch position
    M_inc = int((M / 8) * 0.75)
    N_inc = int((N / 8) * 0.75)

    for i in range(0, M, M_inc):
        for j in range(0, N, N_inc):
            temp = Image[i:i + h, j:j + w]
            c = clahe_clip_cdf_distribute(temp, clip)
            clahe_patch_Img = HE.hist_eq(temp, c)
            blankImage[i:i + h, j:j + w] = clahe_patch_Img

    # cv2 support uint 8 type datatype
    blankImage = blankImage.astype(np.uint8)
    cv2.imshow(name, blankImage)


def Q1c_solver():
    path_lion = "../data/lion.png"
    path_StoneFace = "../data/StoneFace.png"
    clahe_without_overlap(path_lion, "Lion_without_overlapped", 0.1)
    clahe_without_overlap(path_StoneFace, "StoneFace_without_overlapped", 0.4)
    clahe_with_overlap(path_lion, "LionOverlaped", 25, 0.1)
    clahe_with_overlap(path_StoneFace, "StoneFaceOverlapped", 25, 0.4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()