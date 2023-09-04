import numpy as np
import cv2
import matplotlib.pyplot as plt
from numba import jit, double

def filter(img, filt):
  M, N = img.shape
  Mf, Nf = filt.shape
  Mf2 = Mf//2
  Nf2 = Nf//2
  result = np.zeros_like(img)

  for i in range(Mf, M-Mf2):
    for j in range(Nf, N-Nf2):
      window = img[i - Mf2: i + Mf2 + 1, j - Nf2: j + Nf2 + 1]
      center = window[Mf2, Nf2]
      bin_vals = np.zeros((Mf, Nf), dtype=np.uint8)
      
      for ii in range(Mf2):
        for jj in range(Nf2):
          if window[ii, jj] >= center:
            bin_vals[ii, jj] = 1

      bin_vals = bin_vals.flatten()
      lbp_coeff = 0
      for k in range(len(bin_vals)):
        lbp_coeff += bin_vals[k] * (2**k)
      result[i, j] = lbp_coeff
  return result


# img = cv2.imread('face_det/fl.jpg', 0)
# # img = cv2.cvtColor(img, 0)
filt = np.random.random((3,3))
# imgf = filter(img, filt)
# plt.imshow(imgf)
# plt.show()
# img.shape

fast_filter = jit(double[:,:](double[:,:], double[:,:]))(filter)

cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # print(frame.shape)
    print(type(frame))
    frame = fast_filter(frame, filt)
    # print(frame.shape)
    cv2.imshow('frame', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break