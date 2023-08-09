import cv2
from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor()])

# Read image using opencv
i=cv2.imread("/home/kyllex/Pictures/walls/12-Dark.jpg")
print(i.shape)

# Read image using opencv in GrayScale
i_gray=cv2.imread("/home/kyllex/Pictures/walls/12-Dark.jpg", 0)
print(i_gray.shape)

# Read image using pillow
j=Image.open("/home/kyllex/Pictures/walls/12-Dark.jpg")
print(j)
jarray = transform(j)
print(jarray.shape)

# Region selection
# x, y, width, height
r = cv2.selectROI("select area ", i)
print(r)

# Modifying selected Region of interest, r
i[r[1]:r[1] + r[3], r[0]: r[0] + r[2], 0] = 0  # Modifying for each color channel, 0, 1, 2
i[r[1]:r[1]+r[3], r[0]:r[0]+r[2], 1] = 255     # Which is in the BGR format i.e 0-> Blue, 1-> Green, 2->Red
i[r[1]:r[1]+r[3], r[0]:r[0]+r[2], 2] = 230

cv2.imshow("Modified Image", i)
cv2.waitKey(0)

# replacing selected region with different image
k=cv2.imread("/home/kyllex/Pictures/walls/13-Ventura-Dark.jpg")
k1 = cv2.resize(k, (r[2], r[3])) # resize image to width and height of selected region
i[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = k1

cv2.imshow("Image replaced in selected area", i)
cv2.waitKey(0)


# cv2.imshow("Image grayscale", i)
# cv2.waitKey(0)

