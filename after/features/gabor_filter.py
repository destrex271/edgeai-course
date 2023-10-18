# 8 thetha/orientations -> 0, 45, 90, 135 ....
# Implies 40 gabor images from 1 single image

import numpy as np
import matplotlib.pyplot as plt
import cv2

ims = []
fig = plt.figure()
k = 1
for i in range(0, 10):
    g_kernel = cv2.getGaborKernel((21, 21), 8.0, i*np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    img = cv2.imread('dem.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)
    filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB)
    ims.append(filtered_img)
    # cv2.imshow('image', img)
    # cv2.imshow(f'filtered image {i}*pi/4', filtered_img)

    # h, w = g_kernel.shape[:2]
    # g_kernel = cv2.resize(g_kernel, (3*w, 3*h), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('gabor kernel (resized)', g_kernel)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
# for im in ims:
#     plt.subplot(im)
for i in range(len(ims)):
    ax = fig.add_subplot(2, 5, i + 1)
    ax.imshow(ims[i], extent=(0, 900, 0, 900))

fig.tight_layout()
plt.show()

