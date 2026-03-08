import cv2


cap = cv2.VideoCapture(0)

# this will capture initial background frame
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2)

    # Convert to grey scale and blur to remove noise
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    #thresholding to get binary image and dilating to fill in holes, then finding contours
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    #dilated to fill in holes, then finding contours
    dilated = cv2.dilate(thresh, None, iterations=3)

    #find contours of moving objects
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 5000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # It will display the resulting frame
    cv2.imshow('Motion Detector', frame1)

    # Update the frames
    frame1 = frame2
    ret, frame2 = cap.read()

    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release(100000)
cv2.destroyAllWindows()