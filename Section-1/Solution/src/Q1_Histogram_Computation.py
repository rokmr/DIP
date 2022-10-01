import numpy as np
import skimage.io
from matplotlib import pyplot

X = list(range(0, 256))

def histogram_Function(figureName: str, p: str):
    imagePath = p
    Y = []
    image = skimage.io.imread(imagePath)
    for i in range(256):
        Y.append(np.count_nonzero(image == i))
    return Y, image


def Q1_solver():
    print(f" Please wait program is running...")
    Q1a, Img1 = histogram_Function("GulmoharMarg", "../data/GulmoharMarg.png")
    Q1b, Img2 = histogram_Function("GulmoharMargBright", "../data/GulmoharMargBright.png")
    Q1c, Img3 = histogram_Function("GulmoharMargDark", "../data/GulmoharMargDark.png")

    pyplot.subplot(231)
    pyplot.bar(X, Q1a)
    pyplot.title("GulmoharMarg")
    pyplot.subplot(234)
    pyplot.hist(Img1)
    pyplot.title("GulmoharMarg- Inbuilt Fun")

    pyplot.subplot(232)
    pyplot.bar(X, Q1b)
    pyplot.title("GulmoharMargBright")
    pyplot.subplot(235)
    pyplot.hist(Img2)
    pyplot.title("GulmoharMargBright- Inbuilt Fun")

    pyplot.subplot(233)
    pyplot.bar(X, Q1c)
    pyplot.title("GulmoharMargDark")
    pyplot.subplot(236)
    pyplot.hist(Img3)
    pyplot.title("GulmoharMargDark- Inbuilt Fun")

    pyplot.show()