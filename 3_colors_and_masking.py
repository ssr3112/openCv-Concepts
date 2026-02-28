import cv2
import numpy as np

image=cv2.imread('vk1.jpg')

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

#defining blue color range in HSV
lower_blue=np.array([100,150,50])
upper_blue=np.array([130,255,255])


# this turn the pixels within the blue range to white and the rest to black
mask=cv2.inRange(hsv,lower_blue,upper_blue)

result=cv2.bitwise_and(image,image,mask=mask)

cv2.imshow('Original',image)
cv2.imshow('Gray',gray)
cv2.imshow('HSV',hsv)
cv2.imshow('Mask',mask)
cv2.imshow('Result',result)

cv2.waitKey(0)
cv2.destroyAllWindows()