import numpy as np
import cv2

im = cv2.imread('./im.jpg')
im = cv2.cvtColor(im, 0)
r,c,ch = im.shape
print(r, c)
# cv2.imshow('image', im)
for chan in range(ch):
    for row in range(1, r-1):
        for col in range(1, c-1):
            t = im[row - ((r-1)//2): row + ((r-1)//2), col - ((c-1)//2): col + ((c-1)//2), chan]
            print(t.shape)
            # calculate product of elements in numpy array t
            prod = np.prod(t.flatten())
            # convert to uint8
            im[row, col] = np.uint8(prod**(1.0/(9)))
            print("OK", row, col)

cv2.imshow('image', im)
cv2.waitKey(0)