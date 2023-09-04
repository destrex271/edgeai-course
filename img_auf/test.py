import numpy as np
import cv2

def gamma_corr(image, gamma):
    invGamma = 1.0/gamma
    table= np.array([((i/255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def log_transform(image, c):
    table = np.array([(np.log((i/255.0)+1) * c) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def img_add(image, factor):
    return image+factor


def img_mul(image, factor):
    return image*factor


def img_scale_abs(image, alpha, beta):
    return image*alpha + beta

i = cv2.imread("/home/kyllex/Pictures/dreamstime_xxl_65780868_small.jpg")
i_corr1 = gamma_corr(i, 0.6)
i_corr2 = log_transform(i, 1.2)
i_corr3 = img_add(i, 12)
i_corr4 = img_mul(i, 0.0001)
i_corr5 = img_scale_abs(i, 1.04, 12)

cv2.imshow('orig',i)
cv2.waitKey(0)
cv2.imshow('gamma', i_corr1)
cv2.waitKey(0)
cv2.imshow('log', i_corr2)
cv2.waitKey(0)
cv2.imshow('add', i_corr3)
cv2.waitKey(0)
cv2.imshow('mult', i_corr4)
cv2.waitKey(0)
cv2.imshow('mult', i_corr5)
cv2.waitKey(0)
