import cv2
import numpy as np


vid_feed = cv2.VideoCapture(0)
face_detect_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

WIDTH = 800
HEIGHT = 800

q1_image = cv2.imread('/home/kyllex/Pictures/walls/12-Dark.jpg')
q2_image = cv2.imread('/home/kyllex/Pictures/walls/13-Ventura-Dark.jpg')
q3_image = cv2.imread('/home/kyllex/Pictures/walls/12-Dark.jpg')
q4_image = cv2.imread('/home/kyllex/Pictures/walls/13-Ventura-Dark.jpg')

quads = [(0, 0, WIDTH/2, HEIGHT/2), (0, HEIGHT/2, WIDTH/2, HEIGHT), (WIDTH/2, HEIGHT/2, WIDTH, HEIGHT), (WIDTH/2, 0, WIDTH, HEIGHT/2)]

while(True):
    ret, frame = vid_feed.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = face_detect_classifier.detectMultiScale(gray_frame, 1.3, 5)

    for x, y, wid, ht in results:
        print(x, y)
        cv2.rectangle(frame,(x,y),(x + wid, y + ht), (242,0,0), 2)
        up_image = cv2.resize(q1_image, (wid, ht))
        # if x >= quads[0][0] and x <= quads[0][2] and y >= quads[0][1] and y <= quads[0][3]:
        #     up_image = cv2.resize(q1_image, (wid, ht))
        # if x >= quads[1][0] and x <= quads[1][2] and y >= quads[1][1] and y <= quads[1][3]:
        #     up_imahe = cv2.resize(q2_image, (wid, ht))
        # if x >= quads[2][0] and x <= quads[2][2] and y >= quads[2][1] and y <= quads[2][3]:
        #     up_imahe = cv2.resize(q3_image, (wid, ht))
        # if x >= quads[3][0] and x <= quads[3][2] and y >= quads[3][1] and y <= quads[3][3]:
        #     up_imahe = cv2.resize(q4_image, (wid, ht))
        frame[y:y+ht, x:x+wid] = up_image


    # cv2.resizeWindow("Vid feed", 800, 800)
    cv2.imshow('Vid Feed', frame)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

vid_feed.release()
cv2.destroyAllWindows()
