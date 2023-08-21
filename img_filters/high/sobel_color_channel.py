import cv2
import numpy as np

# For edges

vid = cv2.VideoCapture(0)

while(True):

    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, 0)
    # image, datatype, horizontal edges, verticaledges, kernel size
    frame_ede = cv2.Sobel(gray, cv2.CV_8U, 1, 1, ksize=5)
    cv2.imshow('edge', np.uint8(frame_ede))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()