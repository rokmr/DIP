from matplotlib import pyplot as plt
import numpy as np
import math
import skimage
from skimage.morphology import disk

def q2_solver():
    print("---------------Image Denoising:--------------")
    print("            ::Q2a is being solved::          ")

    imageQ2a = skimage.io.imread("./circuitboard.tif")

    #using inbuilt filter to image to create output
    image_median = skimage.filters.median(imageQ2a)
    image_mean = skimage.filters.rank.mean(imageQ2a, footprint=disk(5))

    #plotting the output
    plt.figure()
    plt.subplot(131)
    plt.title("Original Image")
    plt.imshow(imageQ2a, cmap='gray')
    plt.subplot(132)
    plt.title("Mean Filter Image")
    plt.imshow(image_mean,cmap='gray')
    plt.subplot(133)
    plt.title("Median Filter Image")
    plt.imshow(image_median,cmap='gray')
    plt.show()

    print("            ::Q2b is being solved::          ")
    # importing image
    imageQ2b = skimage.io.imread("./noisybook.png")
    img2 = imageQ2b

    # Using hstack and vstack for removing artifacts
    a = imageQ2b[0:3][:]
    imageQ2b = np.vstack((a, a, a, imageQ2b, a, a, a))
    b = imageQ2b[:, 0:1]
    imageQ2b = np.hstack((b, b, b, imageQ2b, b, b, b))
    result_image = np.zeros(imageQ2b.shape, dtype=float)

    # defining global variables
    e = math.exp(1)
    pi = math.pi
    s = 20
    b = 0.05
    r1 = np.zeros(imageQ2b.shape).astype("uint8")

    # Functions for calculating arguments for bilateral computation
    def gauss1(u, v):
        Gs = (1 / (2 * pi * s * s)) * (e ** (-0.5 * (u * u + v * v) / (s * s)))
        return Gs

    def gauss2(intensitydiff):
        Gs = (1 / b * ((2 * pi) ** 0.5)) * (e ** (-0.5 * (intensitydiff * intensitydiff) / (b * b)))
        return Gs

    # Bilateral Filter on (178,260)
    A = []
    B = []
    C = []

    def bilateralfilter2(img, k=178, l=260):
        wb, sum1 = 0, 0
        # filter window is taken as (7*7)
        for m in range(k - 3, k + 4):
            for n in range(l - 3, l + 4):
                A.append(gauss1(k - m, l - n))
                B.append(gauss2(img[m][n] - img[k][l]))
                C.append(gauss1(k - m, l - n) * gauss2(img[m][n] - img[k][l]))

    bilateralfilter2(img2, 178, 260)

    # Plotting P, Gmap, Hmap and Gmap ⊙ Hmap
    plt.subplot(311)
    plt.plot(A)
    plt.ylabel('G')

    plt.subplot(312)
    plt.plot(B)
    plt.ylabel('H')

    plt.subplot(313)
    plt.plot(C)
    plt.ylabel('G 0 H')
    plt.show()

    # Bilateral Filter
    def bilateralfilter(img, k, l):
        wb, sum1 = 0, 0
        # filter window is taken as (7*7)
        for m in range(k - 3, k + 4):
            for n in range(l - 3, l + 4):
                wb = wb + gauss1(k - m, l - n) * gauss2(img[m][n] - img[k][l])
                sum1 = sum1 + img[m][n] * gauss1(k - m, l - n) * gauss2(img[m][n] - img[k][l])
        r = float(sum1 / wb)
        return r

    # function to define the region of filter application here a,b is the height range and y,z is the width range
    def filterapply(img, a, b, y, z):
        for i in range(a, b + 1):
            for j in range(y, z + 1):
                r1[i][j] = bilateralfilter(img, i, j)
        return r1

    # in-build Bilateral filter
    bilat = skimage.restoration.denoise_bilateral(imageQ2b)
    plt.figure()
    plt.subplot(121)
    plt.title("Input Image")
    plt.imshow(imageQ2b, cmap='gray')
    plt.subplot(122)
    plt.title("Output using in-built bilateral filter")
    plt.imshow(bilat, cmap='gray')
    plt.show()

    r1 = filterapply(imageQ2b, 3, 690, 3, 690)
    r1 = r1[3:691, 3:691]

    print("Please wait!!!")
    print("Output using derived bilateral filter")
    # plotting image
    plt.figure()
    plt.subplot(121)
    plt.title("Input Image")
    plt.imshow(imageQ2b, cmap='gray')
    plt.subplot(122)
    plt.title("Output using derived bilateral filter")
    plt.imshow(r1, cmap='gray')
    plt.show()
