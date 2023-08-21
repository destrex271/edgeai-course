# Low pass -> Smoothing
# High Pass -> Edge
# Mask -> Simple windows

# Implement -> 1) mean, max, min, median 2) Weighted Mean 

import imutils
import cv2
import numpy as np
# from PIL import Image,ImageFilter


# i=Image.open('./im.jpg')
# i.filter(ImageFilter.MinFilter(size=25))
# i.save('fil2.jpg')
#implement max,median filters

img = cv2.imread('./im.jpg')

kernel = np.ones((5,5), np.float32)/45

dst = cv2.filter2D(img, -1, kernel)

cv2.imshow("Destination", dst)

cv2.waitKey(0)