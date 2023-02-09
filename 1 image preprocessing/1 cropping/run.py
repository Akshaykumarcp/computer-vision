# pip install opencv-python

# Import packages
import cv2

img = cv2.imread('image preprocessing\cropping\ertiga.png')

print(img.shape) # (280, 420, 3)
cv2.imshow("original", img)
cv2.waitKey(0)

# crop car
x = 67
y = 55
h = 220
w = 360

# Cropping an image
cropped_image = img[y:h, x:w]

# Display cropped image
cv2.imshow("cropped", cropped_image)
cv2.waitKey(0)

# Save the cropped image
# cv2.imwrite("Cropped Image.jpg", cropped_image)

cv2.destroyAllWindows()