import cv2
import numpy as np

lap_kernel = np.matrix([[0,1,0],[1,-4,1],[0,1,0]])

vid_feed = cv2.VideoCapture(0)

while True:
    ret, frame = vid_feed.read()
    frame = cv2.filter2D(frame, -1, lap_kernel*5)

    cv2.imshow('fr', frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

vid_feed.release()
cv2.destroyAllWindows()