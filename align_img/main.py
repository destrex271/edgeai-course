import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('./m.jpg')
rows,cols,ch = img.shape


   

plt.imshow(img)
x = plt.ginput(4)
print(x)
plt.show()
   

pts1 = np.float32([x[0],x[1],x[2],x[3]])
pts2 = np.float32([[0,0],[cols,0],[0,rows],[cols,rows]])

M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img,M,(cols,rows))

plt.subplot(121)
plt.imshow(img)
plt.title('Input')
plt.subplot(122)
plt.imshow(dst)
plt.title('Output')
plt.show()

