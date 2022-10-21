import numpy as np
import math
from sklearn.metrics import mean_squared_error as mse
from matplotlib import pyplot as plt
import skimage

def q3_solver():
    print("------------DFT as matrix products----------")
    print("            ::Q3 is being solved::          ")

    #importing image
    image=skimage.io.imread("./characters.tif")
    M,N=image.shape

    #defining math variable that will be used
    j=complex(0,1)
    pi=math.pi

    #Creating Blank image
    f=np.zeros((M,M))
    m,n=np.indices(f.shape)

    #Defining  Matrix A i.e.,(tilda)
    w=-j*2*pi*m*n/M
    A=np.exp(w)

    #Calculating DFT using matrix as defined above
    F=np.dot(np.dot(A.T,image),A)

    #Taking dft using in-built function
    image_fft=np.fft.fft2(image)

    #plotting the output
    plt.figure()
    plt.subplot(121)
    output_inbuilt=np.log1p(np.abs(image_fft))
    plt.title("Output using inbuilt fft")
    plt.imshow(output_inbuilt,cmap='gray')

    plt.subplot(122)
    output_derived=np.log1p(np.abs(F))
    plt.title("Output using derived fft")
    plt.imshow(output_derived,cmap='gray')
    plt.show()

    #Calculating mean Square Absolute error
    error=mse(output_inbuilt,output_derived)
    print(f"The mean square calvulated using inbuilt : {error}")
    error_derived=np.sum((output_derived-output_inbuilt)**2)/(M*N)
    print(f"The mean square calvulated using inbuilt : {error_derived}")

    res=np.dot(A.conj().T,A) #Plotting A^H * A
    res=np.log1p(np.abs(res))
    plt.figure()
    plt.title("Plot of Ah * A")
    plt.imshow(res,cmap='gray')
    plt.show()
