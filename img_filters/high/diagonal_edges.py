# Kirsch filter

'''
-3 -3 -3
-3  0 -3
 5  5  5
'''
# 8 possible rotations for above kernel

'''
Laplacian

 0 -1  0
-1  4 -1
 0 -1  0
'''

'''
Capture video of people with their faces blurred and surrounding sharpened'''


# Using Kirsh Filter
import cv2
import numpy as np

vid = cv2.VideoCapture(0)

while True:

    ret, frame = vid.read()
    _, _, chan = frame.shape
    for ch in range(chan):
        im = frame[:,:,ch]
        kernel = np.matrix([[-3,-3,-3],[-3,0,-3],[5,5,5]])
        edge = cv2.filter2D(im, -1, kernel)
        frame[:,:,ch] += edge

    cv2.imshow("Image", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()