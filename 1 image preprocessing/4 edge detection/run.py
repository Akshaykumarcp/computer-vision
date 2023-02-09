# https://pyimagesearch.com/2021/05/12/opencv-edge-detection-cv2-canny/
# https://dontrepeatyourself.org/post/edge-and-contour-detection-with-opencv-and-python/

import cv2

# load the image, convert it to grayscale, and blur it slightly
# image = cv2.imread('image preprocessing/4 edge detection/ertiga.png')
# image = cv2.imread('image preprocessing/4 edge detection/cottonbro.jpg')
image = cv2.imread('image preprocessing/4 edge detection/scorpio_front.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# show the original and blurred images
cv2.destroyAllWindows()

cv2.imshow("Original", image)
cv2.imshow("Blurred", blurred)

# compute a "wide", "mid-range", and "tight" threshold for the edges
# using the Canny edge detector
wide = cv2.Canny(blurred, 10, 200)
mid = cv2.Canny(blurred, 30, 150)
tight = cv2.Canny(blurred, 240, 250)

test = cv2.Canny(blurred, 100, 200, apertureSize=5)

# show the output Canny edge maps
# cv2.imshow("Wide Edge Map", wide)
# cv2.imshow("Mid Edge Map", mid)
# cv2.imshow("Tight Edge Map", tight)
# cv2.imshow("Test Map", test)
cv2.waitKey(0)

""" # Find contour and sort by contour area
cnts = cv2.findContours(test, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

# Find bounding box and extract ROI
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    ROI = image[y:y+h, x:x+w]
    break

cv2.imshow('ROI',ROI)
# cv2.imwrite('ROI.png',ROI)
cv2.waitKey()
 """
# define a (3, 3) structuring element
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# # apply the dilation operation to the edged image
# dilate = cv2.dilate(mid, kernel, iterations=1)

# find the contours in the edged image
contours, _ = cv2.findContours(mid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

image_copy = image.copy()
# draw the contours on a copy of the original image
cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)
print(len(contours), "objects were found in this image.")

cv2.imshow("Edged image", mid)
cv2.imshow("contours", image_copy)
cv2.waitKey(0)