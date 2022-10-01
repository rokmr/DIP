import cv2
import FSCS
import numpy as np

# Importing Image
path_lion = "../data/lion.png"
path_Hazy = "../data/Hazy.png"
path_StoneFace = "../data/StoneFace.png"


def pdf_cdf_calculator(Image):
    """This function is used to calculate pdf and cdf of intensity for the given image"""
    M, N = Image.shape
    Y = []

    # Getting intensity values
    for i in range(256):
        Y.append(np.count_nonzero(Image == i))
    Y = np.array(Y)

    # Calculating pdf
    pdf = Y / (M * N)

    # Calculating cdf
    cdf = np.zeros(shape=(256,))
    cdf[0] = pdf[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + pdf[i]

    return pdf, cdf


def hist_eq(Image, cdf_image):
    """This function is used to generate histogram equalized Image for given Image And its cdf"""
    # Reading Image
    M, N = Image.shape

    # Creating a black Image of same size
    HE_Image = np.zeros(shape=(M, N))

    # putting values of cdf for corresponding intensity level
    for i in range(M):
        for j in range(N):
            HE_Image[i][j] = cdf_image[Image[i][j]] * 255
    HE_Image = HE_Image.astype(np.uint8)
    return HE_Image


def plotting_Img_ImgHE(Image, HE_Image, name: str):
    n = name + "_HE_Image"
    cv2.imshow(name, Image)
    cv2.imshow(n, HE_Image)


def Q1_b_HE(path_img, name: str):
    Img = FSCS.import_image(path_img)
    pdf_image, cdf_image = pdf_cdf_calculator(Img)
    Img_HE = hist_eq(Img, cdf_image)
    plotting_Img_ImgHE(Img, Img_HE, name)


def Q1b_solver():
    Q1_b_HE(path_lion, "lion")
    Q1_b_HE(path_Hazy, "Hazy")
    Q1_b_HE(path_StoneFace, "StoneFace")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
