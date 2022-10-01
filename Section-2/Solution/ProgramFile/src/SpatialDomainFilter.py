import cv2
import FSCS

scalingfactor = 2
path2 = "../data/blur.png"
path1 = "../data/noisy.png"


def highBoostFilter(path, name: str):
    """ This funtion will take path of image and and reun the high boost image i.e., sharped version"""
    image = FSCS.import_image(path)
    cv2.imshow(name, image)

    # Using GaussiaBlur Filter to blur the image
    gaussianBlur_Image = cv2.GaussianBlur(image, (5, 5), 0)

    masked_Image = image - gaussianBlur_Image

    sharpImage = image + scalingfactor * masked_Image

    outputName = name + "sharpedImage"
    cv2.imshow(outputName, sharpImage)


def Q3_solver():
    highBoostFilter(path1, "Blur")
    highBoostFilter(path2, "Noisy")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
