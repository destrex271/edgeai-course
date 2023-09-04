import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./m.jpg')
row, cols, ch = img.shape
im2 = cv2.imread('./ma.jpeg')
# gray = cv2.cvtColor(img, 0)
# print(gray.shape)
face_detection = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

gray = cv2.cvtColor(im2, 0)
detect = face_detection.detectMultiScale(gray, 1.1, 6)
print(detect)

plt.imshow(gray)
a,b,c,d = plt.ginput(4)
print(a)
print("FACE", detect)
# a,b,c,d = detect
plt.show()
pts1 = np.float32([a,b,c,d])
pts2 = np.float32([[0,0],[cols,0],[0,row],[cols,row]])
'''MTCNN'''
M =cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(gray, M, (cols, row))

plt.subplot(121)
plt.imshow(img)
plt.title('Input')
plt.subplot(122)
plt.imshow(dst)
plt.title('Output')
plt.show()

