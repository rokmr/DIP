import numpy as np
import math
from matplotlib import pyplot as plt
import skimage

def fscs(image):
    '''Funtion will take image and return the Full Scale Contrast Streched Image'''
    a = np.amin(image)
    b = np.amax(image)
    constant = 255 / (b - a)
    ImageFSCS = constant * (image - a)
    ImageFSCS = ImageFSCS.astype(np.uint8)
    return ImageFSCS


def fft_image(image):
    '''Function retun center shifted FFT'''
    image_fft = np.fft.fft2(image)  # performing FFT on image
    image_fft_shift = np.fft.fftshift(image_fft)  # Making FFT centered
    return image_fft_shift


def filter_image(fft_shifted, filter):
    '''Tis function will take center shifted fft image and a filter(in FD) which need to applied and return the Spce
    domain output '''
    filtered_fft = fft_shifted * filter  # Applying filter in Frequency domain
    image_fft_ishift = np.fft.ifftshift(filtered_fft)  # Shifting image from centered frequency back to original
    output = np.abs(np.fft.ifft2(image_fft_ishift))  #inverse fft and takingabsolute value
    return output, filtered_fft


def lowPass_filter_generator(image, D0):
    '''This function will generate the ILPF of same size as given image and of given cutoff frequency'''
    P, Q = image.shape
    u, v = np.indices(image.shape)
    D = np.sqrt((u - P / 2) ** 2 + (v - Q / 2) ** 2)
    ILPF = D < D0
    ILPF = ILPF * 1
    return ILPF

def q1_solver():
    print("---------Frequency Domain Filterring---------")
    print("            ::Q1a is being solved::          ")

    # Variable declaration
    M = 501
    N = 501
    ua = 40
    va = 60
    ub = 20
    vb = 100
    pi = math.pi

    w1a = 2 * pi * ua / M
    w2a = 2 * pi * va / N

    w1b = 2 * pi * ub / M
    w2b = 2 * pi * vb / N

    shape = M, N
    m, n = np.indices(shape)

    # Generating Image As specified
    image_A = np.sin(w1a * m + w2a * n)
    image_B = np.sin(w1b * m + w2b * n)
    image_AB = image_A + image_B

    FFT_A = fft_image(image_A)
    FFT_B = fft_image(image_B)

    FFT_AB = FFT_A + FFT_B
    image_ifft_AB = np.fft.ifft2(FFT_AB)  # taking Inverse FFt of image

    # Plotting Image
    plt.figure()
    plt.subplot(121)
    plt.title("Image A")
    plt.imshow(image_A, cmap='gray')
    plt.subplot(122)
    plt.title("Image A FFT")
    plt.imshow(np.log1p(np.abs(FFT_A)), cmap='gray')

    plt.figure()
    plt.subplot(121)
    plt.title("Image B")
    plt.imshow(image_B, cmap='gray')
    plt.subplot(122)
    plt.title("Image B FFT")
    plt.imshow(np.log1p(np.abs(FFT_B)), cmap='gray')

    plt.figure()
    plt.subplot(121)
    plt.title("Image with point wise addition")
    plt.imshow(image_AB, cmap='gray')
    plt.subplot(122)
    plt.title("IFFT image of point wise sum of FFT A & FFT B")
    plt.imshow(np.log1p(np.abs(image_ifft_AB)), cmap='gray')
    plt.show()

    #Q1b
    print("            ::Q1b is being solved::          ")

    imageQ1b = skimage.io.imread("./dynamicSine.png")
    image_fft = np.fft.fft2(imageQ1b)
    image_fft_shift = np.fft.fftshift(image_fft)

    # Low pass filter D=20
    ILPF20 = lowPass_filter_generator(imageQ1b, 20)

    # High Pass Filter D=60
    ILPF60 = lowPass_filter_generator(imageQ1b, 60)
    IHPF60 = 1 - ILPF60

    # Band Pass filter D=20,40    40,60
    ILPF40 = lowPass_filter_generator(imageQ1b, 40)
    IHPF20 = 1 - ILPF20
    IBPF1 = IHPF20 * ILPF40

    IHPF40 = 1 - ILPF40
    IBPF2 = ILPF60 * IHPF40

    # Generating output corresponding to filter
    ILPF_output, _ = filter_image(image_fft_shift, ILPF20)
    IHPF_output, _ = filter_image(image_fft_shift, IHPF60)
    IBPF1_output, _ = filter_image(image_fft_shift, IBPF1)
    IBPF2_output, _ = filter_image(image_fft_shift, IBPF2)

    # Output plotting
    plt.figure()
    plt.title("Input Image")
    plt.imshow(imageQ1b, cmap='gray')

    plt.figure()
    plt.subplot(221)
    plt.title("Output with ILPF , D0=20")
    plt.imshow(ILPF_output, cmap='gray')
    plt.subplot(222)
    plt.title("Output with IHPF , D0=60")
    plt.imshow(IHPF_output, cmap='gray')
    plt.subplot(223)
    plt.title("Output with IBPF1 , D=(20,40)")
    plt.imshow(IBPF1_output, cmap='gray')
    plt.subplot(224)
    plt.title("Output with IBPF2 , D=(40,60)")
    plt.imshow(IBPF2_output, cmap='gray')
    plt.show()

    print("            ::Q1c is being solved::          ")

    imageQ1c = skimage.io.imread("./characters.tif")
    image_fft = np.fft.fft2(imageQ1c)
    image_fft_shift = np.fft.fftshift(image_fft)
    P, Q = imageQ1c.shape
    D0 = 100
    u, v = np.indices(imageQ1c.shape)

    # Forming Gaussian Filter and applying on image
    D = np.sqrt((u - P / 2) ** 2 + (v - Q / 2) ** 2)
    GLPF = np.exp(-D ** 2 / (2 * D0 ** 2))
    output_image, filtered = filter_image(image_fft_shift, GLPF)

    # Plotting Image
    plt.figure()
    plt.subplot(231)
    plt.title("Input Image")
    plt.imshow(imageQ1c, cmap='gray')
    plt.subplot(232)
    plt.title("FFT of image")
    plt.imshow(np.log1p(np.abs(image_fft)), cmap='gray')
    plt.subplot(233)
    plt.title("Centered FFT of image")
    plt.imshow(np.log1p(np.abs(image_fft_shift)), cmap='gray')
    plt.subplot(234)
    plt.title("Gaussian filter in Frequency Domain")
    plt.imshow(GLPF, cmap='gray')
    plt.subplot(235)
    plt.title("Filtered image in frequency domain")
    plt.imshow(np.log1p(np.abs(filtered)), cmap='gray')
    plt.subplot(236)
    plt.title("Filtered image in Space Domain")
    plt.imshow(output_image, cmap='gray')
    plt.show()

