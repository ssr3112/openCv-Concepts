import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# STEP 1: Capture the "Reference" frame (The empty room)
# We wait a second to let the camera adjust its brightness
for i in range(30):
    ret, static_back = cap.read()

# Convert reference to gray and blur once
static_back = cv2.cvtColor(static_back, cv2.COLOR_BGR2GRAY)
static_back = cv2.GaussianBlur(static_back, (21, 21), 0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # Create a copy to draw on
    output = frame.copy()

    # Step 2: Process current frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Step 3: Compare CURRENT frame to the STATIC background
    diff = cv2.absdiff(static_back, gray)
    
    # Step 4: Binary Threshold (Adjust '25' if it's too sensitive)
    _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
    
    # Step 5: Dilate to join the white "motion" pixels together
    dilated = cv2.dilate(thresh, None, iterations=4)

    # Step 6: Find and Draw Motion
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    movement_detected = False
    for cnt in contours:
        if cv2.contourArea(cnt) < 3000: # Ignore tiny movements
            continue
        
        movement_detected = True
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if movement_detected:
        cv2.putText(output, "ALARM: MOVEMENT", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Show the magic happening in 3 windows
    cv2.imshow("1. Real Feed", output)
    cv2.imshow("2. Difference (Heatmap)", diff)
    cv2.imshow("3. Binary Mask (Logic)", dilated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


    # Press 'r' to reset the background (if light changes)
    if key == ord('r'):
        static_back = gray

cap.release()
cv2.destroyAllWindows()
