from PIL import Image, ImageFilter

im = Image.open('./im.jpg')

filt_img = im.filter(ImageFilter.MedianFilter(15))
filt_img.show()

filt_img = im.filter(ImageFilter.MinFilter(15))
filt_img.show()

filt_img = im.filter(ImageFilter.MaxFilter(3))
filt_img.show()

filt_img = im.filter(ImageFilter.BoxBlur(15))
filt_img.show()

filt_img = im.filter(ImageFilter.GaussianBlur(15))
filt_img.show()

filt_img = im.filter(ImageFilter.EDGE_ENHANCE)
filt_img.show()

filt_img = im.filter(ImageFilter.EDGE_ENHANCE_MORE)
filt_img.show()

# Gaussian Blur is same as SMOOTH filter run over 10000 times
# filt_img = im
# for _ in range(0, 10000):
#     filt_img = filt_img.filter(ImageFilter.SMOOTH)
# filt_img.show()


# Max, Mean -> Image dependent
# Mean, median, weighted -> always smoothing
# Sobel, prewit -> Edge

filt_img = im
filt_img = filt_img.filter(ImageFilter.EMBOSS)
filt_img.show()

# filt_img = im.filter(ImageFilter.UnsharpMask)
# filt_img.show()
