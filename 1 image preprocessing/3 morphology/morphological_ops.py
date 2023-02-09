# pip install opencv-contrib-python
# https://pyimagesearch.com/2021/04/28/opencv-morphological-operations/

import cv2

image = cv2.imread("image preprocessing/3 morphology/small.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# EROSION

cv2.imshow("Original", image)
# apply a series of erosions
for i in range(0, 10):
	eroded = cv2.erode(gray.copy(), None, iterations=i + 1)
	cv2.imshow("Eroded {} times".format(i + 1), eroded)
	cv2.waitKey(0)

 # DILATION

 # apply a series of dilations
for i in range(0, 10):
	dilated = cv2.dilate(gray.copy(), None, iterations=i + 1)
	cv2.imshow("Dilated {} times".format(i + 1), dilated)
	cv2.waitKey(0)

# OPENING

cv2.destroyAllWindows()
cv2.imshow("Original", image)
kernelSizes = [(3, 3), (5, 5), (7, 7)]
# loop over the kernels sizes
for kernelSize in kernelSizes:
	# construct a rectangular kernel from the current size and then
	# apply an "opening" operation
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
	opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
	cv2.imshow("Opening: ({}, {})".format(
		kernelSize[0], kernelSize[1]), opening)
	cv2.waitKey(0)

# CLOSING

# close all windows to cleanup the screen
cv2.destroyAllWindows()
cv2.imshow("Original", image)
# loop over the kernels sizes again
for kernelSize in kernelSizes:
	# construct a rectangular kernel form the current size, but this
	# time apply a "closing" operation
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
	closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
	cv2.imshow("Closing: ({}, {})".format(
		kernelSize[0], kernelSize[1]), closing)
	cv2.waitKey(0)

# Morfological gradient

# close all windows to cleanup the screen
cv2.destroyAllWindows()
cv2.imshow("Original", image)
# loop over the kernels a final time
for kernelSize in kernelSizes:
	# construct a rectangular kernel and apply a "morphological
	# gradient" operation to the image
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
	gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
	cv2.imshow("Gradient: ({}, {})".format(
		kernelSize[0], kernelSize[1]), gradient)
	cv2.waitKey(0)

