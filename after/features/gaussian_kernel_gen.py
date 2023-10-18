import cv2
import numpy as np


def gaussian_kernel(size=3, u=0, sigma=1):

    x = np.linspace(-1, 1, num=size, axis=0)
    y = np.linspace(-1, 1, num=size, axis=0)
    x, y = np.meshgrid(x, y)
    dst = np.sqrt((x-u)**2 + (y-u)**2)
    div = 1/(2*np.pi*sigma**2)
    kernel = np.exp(-((dst-u)**2 / (2*sigma**2))) * div
    return kernel/np.sum(kernel)


kern = gaussian_kernel(15)
image = cv2.imread("dem.jpg")
filt = cv2.filter2D(image, ddepth=-1, kernel=kern)

cv2.imshow("IM", filt)
cv2.waitKey(0)
cv2.destroyAllWindows()
