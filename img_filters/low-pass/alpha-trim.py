import numpy as np
import cv2

i = cv2.imread('./im.jpg')
h,w,ch = i.shape

m=3
n=3
d=2

j=np.zeros_like()