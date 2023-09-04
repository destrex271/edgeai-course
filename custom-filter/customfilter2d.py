import numpy
from time import time

def filter2d(image, filt):
    M, N = image.shape
    Mf, Nf = filt.shape
    Mf2 = Mf//2   # Mf // 2 to -(Mf // 2 + 1)
    Nf2 = Nf // 2
    result  = numpy.zeros_like(image)
    # Mf2 -> M - Mf2 to avoid crossing over boundaries
    for i in range(Mf2, M-Mf2):
        for j in range(Nf2, N-Nf2):
            result[i,j] = numpy.sum(image[i-Mf2:i+Mf2+1, j-Nf2:j+Nf2+1] * filt)

    return result

tic = time()
image = numpy.random.random((100, 100))
filt = numpy.random.random((9, 9))
res = filter2d(image, filt)
toc = time()
print('Time taken: ', toc-tic)
