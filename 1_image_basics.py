import cv2
import numpy as np

image = cv2.imread('vk1.jpg') 

if image is None:
    image = np.zeros((512, 512, 3), dtype="uint8")
    cv2.rectangle(image, (100, 100), (400, 400), (0, 255, 0), -1)

print(type(image))
print(image.shape)
print(image.size)
print(image.dtype)

pixel_value = image[100, 100]
print(pixel_value)

cv2.imshow('Main Window', image)
cv2.imwrite('output_test.jpg', image)

cv2.waitKey(0)
cv2.destroyAllWindows()