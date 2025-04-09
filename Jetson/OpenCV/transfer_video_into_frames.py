# install first: pip install opencv-python
import cv2
import numpy as np
import os
import pandas as pd


video = cv2.VideoCapture("VID_20250401_122228.mp4")  # Will be updated with the captured video
output_folder = "C:/Storage/Studies/Lapland_AMK/4_semester/Robotics_project/OpenCV/Frames"  # Will be also updated to save on Jetson
os.makedirs(output_folder, exist_ok=True)

i = 0
while True:
    ret, frame = video.read()
    if not ret:
        break
    cv2.imwrite(f"{output_folder}/{i:04d}.jpg", frame)
    i += 1
    cv2.imshow('Extracted Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
