# importing the library
import skimage
import cv2
import numpy as np
import practice
from matplotlib import pyplot

path= "../data/DoubleColorShapes.png"
shapeImage=cv2.imread(path)
grayImage=cv2.cvtColor(shapeImage, cv2.COLOR_BGR2GRAY)

ep=1


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i

    return num

c=[]
d={}
for thresh in range(0, 256, ep):
    print(f"Please Wait:{(256 - thresh) / ep}")
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
    c.append(l)

print(c)
f=most_frequent(c)
t=c[f]
print(f"The threshold:{t}")
print(f"The number of figure :{f}")
e,index=practice.max_sequence_element(c)
print(f"Element:{e}")
print(f"Index:{index}")
