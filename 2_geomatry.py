import cv2
import numpy as np

image = cv2.imread('vk1.jpg')

if image is None:
    image = np.zeros((512, 512, 3), dtype="uint8")
    cv2.putText(image, "Dummy Image", (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

height, width = image.shape[:2]

resized = cv2.resize(image, (width // 2, height // 2))

roi = image[100:400, 200:500]

flipped = cv2.flip(image, 1)
rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

cv2.imshow("Original", image)
cv2.imshow("Resized (Smaller)", resized)
cv2.imshow("Cropped (ROI)", roi)
cv2.imshow("Flipped (Horizontal)", flipped)
cv2.imshow("Rotated (90° Clockwise)", rotated)

cv2.waitKey(0)
cv2.destroyAllWindows()