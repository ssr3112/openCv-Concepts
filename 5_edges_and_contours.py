import cv2
import numpy as np

image = cv2.imread('vk1.jpg')

if image is None:
    image = np.zeros((512, 512, 3), dtype="uint8")
    cv2.putText(image, "Dummy Image", (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 30, 150)
blurred = cv2.GaussianBlur(gray, (21, 21), 0)

kernel = np.ones((5,5), np.uint8)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# finding shapes
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    if cv2.contourArea(cnt) < 1000:
        continue
    # localization
    x, y, w, h = cv2.boundingRect(cnt)

    # drawing green outline for shape
    cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)
    

    # red line for bounding box
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Syntax: putText(image, "Text", (x, y-10), font, scale, color, thickness)
    label="Object Detected"
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


cv2.imshow("Detected Shapes with label", image)
cv2.imshow("Edges", edges)


cv2.waitKey(0)
cv2.destroyAllWindows()




