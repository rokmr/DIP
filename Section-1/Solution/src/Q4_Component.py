# importing the library
import skimage
import cv2
import numpy as np
from matplotlib import pyplot

def Q4_solver():
    path="../data/Shapes.png"
    shapeImage=cv2.imread(path)
    grayImage=cv2.cvtColor(shapeImage, cv2.COLOR_BGR2GRAY)

    thresh = int(skimage.filters.threshold_otsu(grayImage))

    # Generating Binary Image
    binaryImage = grayImage > thresh
    binaryImage = binaryImage * 1
    M, N = binaryImage.shape

    # Component Extraction
    k=1
    R= np.zeros([M,N],dtype=int)
    I=binaryImage
    d={}


    for i in range(M):
        for j in range(N):
            U = binaryImage[i - 1, j]
            L = binaryImage[i, j - 1]
            I = binaryImage[i,j]

            if I == 1 and U==0 and L == 0:
                R[i,j]=k
                k+=1

            elif I == 1 and U==1 and L == 0:
                R[i,j]=R[i-1,j]

            elif I == 1 and U==0 and L == 1:
                R[i,j]=R[i,j-1]

            elif I == 1 and U==1 and L == 1:
                R[i, j] = R[i - 1, j]
                if R[i-1, j] in  list(d.keys()):
                    d.update({R[i, j-1]:d[R[i - 1, j]]})
                else:
                    d.update({R[i, j - 1]: R[i - 1, j]})

    s=set(d.values())
    l=len(s)
    print(f"No. of Shapes: {l}")

    v=list(d.keys())
    for j in range(N):
        for i in range(M):
            if R[i,j] in v:
                R[i,j]=d[R[i,j]]

    l=R.tolist()

    pixelBinvalues=[]
    for item in list(s):
        pixelBinvalues.append(np.count_nonzero(R == item))

    countCircle=0
    for item in pixelBinvalues:
        m=min(pixelBinvalues)
        if item <= m*1.05 and item >= m*0.95:
            countCircle+=1

    print(f"Number of Circle:{countCircle}")