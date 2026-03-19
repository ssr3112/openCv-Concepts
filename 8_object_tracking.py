import cv2

tracker = cv2.TrackerKCF.create()
cap=cv2.VideoCapture('vk1.mp4')

ret, frame=cap.read()
bbox=cv2.selectROI(frame,False)
tracker.init(frame,bbox)

while True:
    ret, frame=cap.read()
    if not ret:break

    seccess, bbox=tracker.update(frame)

    # if tracking is successful draw rectangle
    if seccess:
        x,y,w,h=[int(i) for i in bbox]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,"Tracking",(100,100),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    else:
        cv2.putText(frame,"Lost",(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    cv2.imshow("Tracking",frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
