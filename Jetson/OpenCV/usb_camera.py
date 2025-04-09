import cv2
import numpy as np

# code to get captured video:
camera = cv2.VideoCapture(0)

if (camera.isOpened()):
    print("The Camera is successfully opened")
else:
    print("Could not open the camera")

while True:
    
    success, frame=camera.read()

    if not success:
        print("Not able to read the frame. End.")
        break

    frame2 = cv2.GaussianBlur(frame, (31,31), 0)

    combinedImages = np.hstack((frame, frame2))

    cv2.imshow("Camera video", frame)

    if cv2.waitKey(1) == ord("c"):
        break

camera.release()

cv2.destroyAllWindows()