import cv2
import numpy as np


video = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('background.avi', video, 20.0, (640, 480))
capture = cv2.VideoCapture(0)

image = cv2.imread('temple.jpg')
image = cv2.resize(image, (640, 480))

for i in range(0,60):
    ret, frame = capture.read()
    
while(capture.isOpened()):
    ret, frame = capture.read()
    if not ret:
        break
    frame = np.flip(frame, axis=1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    u_black = np.array([104,153,70])
    l_black = np.array([30,30,0])
    mask1 = cv2.inRange(hsv, l_black, u_black)
    res= cv2.bitwise_and(frame, frame, mask=mask1)
    f=frame-res
    f= np.where(f==0, image, f)
    cv2.imshow("capture", frame)
    cv2.imshow("background magic", f)

    cv2.waitKey(1)
capture.release()
cv2.destroyAllWindows()