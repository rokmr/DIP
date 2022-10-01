# import Library
import skimage
import numpy as np
from matplotlib import pyplot


def m_var_Calculator (C, a, sumC):
    if sumC != 0:
        m = sum(C * a) / sumC
        var = sum(((a - m) ** 2) * C) / sumC

    else:
        m = 0
        var = 0

    return m, var


def Q2_solver():
    # import image and getting its parameters
    imagePath = "../data/coins.png"
    image = skimage.io.imread(imagePath)
    M, N = image.shape

    # Printing Threshold using inbuilt function
    thresh = skimage.filters.threshold_otsu(image)
    print(f"In-built thresh:{thresh}")

    # Getting Frequencies for corresponding intensity level
    H = []
    for i in range(256):
        H.append(np.count_nonzero(image == i))
    H = np.array(H)

    # Setting up variable
    min_within = float("inf")
    max_withinB = 0

    T_Within_init = 0
    T_Between_init = 0

    Yw=[]
    Yb=[]
    X=list(range(256))

    for t in range(0, 256):
        C0 = H[0:t + 1]
        C1 = H[t + 1:256]

        a0 = np.array(range(0, t + 1))
        a1 = np.array(range(t + 1, 256))

        sumC0 = sum(C0)
        sumC1 = sum(C1)

        m0, var0 = m_var_Calculator(C0, a0, sumC0)
        m1, var1 = m_var_Calculator(C1, a1, sumC1)

        mT = m0 * (sumC0 / (M * N)) + m1 * (sumC1 / (M * N))

        varW = (sumC0 * var0 + sumC1 * var1) / (M * N)
        varB = ((sumC0 * ((m0 - mT) ** 2)) + (sumC1 * ((m1 - mT) ** 2))) / (M * N)

        Yw.append(varW)
        Yb.append(varB)

        if varW < min_within:
            T_Within_init = t
            min_within = varW

        if varB != 0 and varB > max_withinB:
            T_Between_init = t
            max_withinB = varB

    print(f"Within thresh:{T_Within_init}")
    print(f"Between thresh:{T_Between_init}")

    #Output Image
    pyplot.subplot(121)
    pyplot.title("Within Variance Plot")
    pyplot.plot(X,Yw)
    pyplot.subplot(122)
    pyplot.title("Between Variance Plot")
    pyplot.plot(X,Yb)
    pyplot.show()


    out_image = image > thresh
    out_image = out_image * 255
    pyplot.title("Otsu's Binarization")
    pyplot.imshow(out_image, cmap='gray')
    pyplot.show()

    print(f"Minimun of within variance is located at:{Yw.index(min(Yw))}")
    print(f"Maximum of between variance is located at:{Yb.index(max(Yb))}")