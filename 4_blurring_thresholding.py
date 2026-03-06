import cv2
import numpy as np

image = cv2.imread('vk1.jpg')

if image is None:
    image = np.zeros((512, 512, 3), dtype="uint8")
    cv2.putText(image, "Step 4 Test", (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


# converting to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# applying Gaussian Blur(size in odd like 3*3,5*5,7*7)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# thresholding any value > 127 become white else 0-black
ret, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

# adaptive thresholding(better for uneven light)
thresh_adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

cv2.imshow("Original", image)
cv2.imshow("Grayscale", gray)
cv2.imshow("Blurred", blurred)
cv2.imshow("Thresholded", thresh)
cv2.imshow("Adaptive Thresholded", thresh_adaptive)

cv2.waitKey(0)
cv2.destroyAllWindows()