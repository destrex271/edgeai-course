import cv2
import numpy as np

vid_feed=  cv2.VideoCapture(0)
face_detect_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while(True):
    ret, frame=vid_feed.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_edge = cv2.Sobel(gray, cv2.CV_8U, 0, 1, ksize=3)
    sharp_img = np.zeros_like(frame)
    for ch in range(3):
        sharp_img[:, :, ch] = frame[:, :, ch] + frame_edge

    detections = face_detect_classifier.detectMultiScale(gray, 1.1, 6)
    for (x, y, w, h) in detections:
        image = cv2.rectangle(sharp_img, (x, y), (x + w, y + h), (0, 255, 0), 4)
        image[y: y+ h, x: x+ w] = cv2.medianBlur(image[y: y+ h, x: x+ w], 13)
    cv2.imshow('frame_edge', sharp_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid_feed.release()
cv2.destroyAllWindows()