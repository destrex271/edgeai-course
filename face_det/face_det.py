import numpy as np
import cv2
import matplotlib.pyplot as plt
from numba import jit, double, uint8, njit, prange

from time import time
# @njit(parallel=True)



# img = cv2.imread('face_det/fl.jpg', 0)
# # img = cv2.cvtColor(img, 0)
filt = np.random.random((3,3))
# imgf = filter(img, filt)
# plt.imshow(imgf)
# plt.show()
# img.shape

# fast_filter = jit(double[:,:](uint8[:,:], uint8[:,:]))(filter)

@njit(parallel=True)
def filter(img, filt):
  M, N = img.shape
  Mf, Nf = filt.shape
  Mf2 = Mf//2
  Nf2 = Nf//2
  result = np.zeros_like(img)

  for i in prange(Mf, M-Mf2):
    for j in prange(Nf, N-Nf2):
      window = img[i - Mf2: i + Mf2 + 1, j - Nf2: j + Nf2 + 1]
      center = window[Mf2, Nf2]
      bin_vals = np.zeros((Mf, Nf), dtype=np.uint8)

      # LBP filter for 3x3 window
      for ii in prange(Mf):
        for jj in prange(Nf):
          if window[ii, jj] >= center:
            bin_vals[ii, jj] = 1

      bin_vals = bin_vals.flatten()
      # print(bin_vals)
      lbp_coeff = 0
      for k in range(len(bin_vals)):
        lbp_coeff += bin_vals[k] * (2**k)
        # print(2**k, bin_vals[k])
      result[i, j] = np.uint8(lbp_coeff)
      # print(result[i, j])
  return result


cam = cv2.VideoCapture(0)

k = 0
b = False

# while k < 10:
#     ret, frame = cam.read()

#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame = filter(frame, filt)
#     cv2.imwrite(f'face_det/data/fl{k}.jpg', frame)
#     k+=1
#     print(k)


while True:
  tic = time()
  ret, frame = cam.read()
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  frame = filter(frame, filt)
  toc = time()
  print("OK")
  cv2.imshow('frame', frame)
  # if not b:
  print(toc - tic)
    # b = True
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

  # cv2.imwrite('face_det/fl.jpg', frame)

  # img = cv2.imread('face_det/fl.jpg', 0)
  # imgf = filter(img, filt)
  # plt.imshow(imgf, cmap="gray")
  # plt.show()
  # cv2.imwrite(f'face_det/data/fl{k}.jpg', imgf)
  # k += 1
print(toc-tic)
