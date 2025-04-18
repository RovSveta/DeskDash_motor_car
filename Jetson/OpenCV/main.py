# needed installations:
# pip install opencv-python
# pip install numpy
# pip install pandas
# pip install torch torchvision torchaudio - PyTorch + TorchVision – for YOLOv5 model inference
# pip install matplotlib seaborn tqdm - YOLOv5 dependencies via ultralytics repo (used with torch.hub.load)
# git clone https://github.com/ultralytics/yolov5 - this is needed to run Yolov5
# cd yolov5
# pip3 install -r requirements.txt
# pip install joblib - Joblib – for loading the weather prediction model:

import cv2
import numpy as np
import pandas as pd
import os
import torch

# this is needed to avoid PosixPath error on Windows, in Linux this is not needed:
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


# Open USB camera:
#camera = cv2.VideoCapture(0)
#camera = cv2.VideoCapture("VID_20250401_122228.mp4")
camera = cv2.VideoCapture("C:/Storage/Studies/Lapland_AMK/4_semester/Robotics_project/DeskDash_motor_car/Jetson/OpenCV/Object_detection.mp4")



if not camera.isOpened():
    print("Failed to open video.")
    exit()

# Adjusted HSV Ranges for detecting red and blue lane lines
# Red
red_lower1 = np.array([0, 70, 70])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([160, 70, 70])
red_upper2 = np.array([179, 255, 255])

# Blue 
blue_lower = np.array([85, 40, 40])
blue_upper = np.array([135, 255, 255])

# this function detects red and blue lines contours, calculates lanes center, calculates average x-positions
# and calculates how far the car (assumed to be in the center of the frame) is from each line:
def compute_center_line_and_distances(mask_red, mask_blue, frame, y_focus_area):
    red_x = []
    blue_x = []

    # find red and blue contours:
    red_line_edges, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_line_edges, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Loop through contours and collect x-values for red lane:
    for edge in red_line_edges:
        if cv2.contourArea(edge) > 100:
            for point in edge:
                x, y = point[0]
                if y >= y_focus_area:
                    red_x.append(x)

    # Loop through contours and collect x-values for red lane:
    for edge in blue_line_edges:
        if cv2.contourArea(edge) > 100:
            for point in edge:
                x, y = point[0]
                if y >= y_focus_area:
                    blue_x.append(x)

    if red_x and blue_x:
        # These represent the average horizontal position of each lane line:
        avg_red = int(np.mean(red_x))
        avg_blue = int(np.mean(blue_x))

        # middle between red and blue → the desired driving path:
        center_x = (avg_red + avg_blue) // 2
        
        # this is a car position (assumed position), center of the screen
        height, width = frame.shape[:2]
        car_position_x = width // 2

        # draw a sentre line:
        cv2.line(frame, (center_x, 0), (center_x, frame.shape[0]), (0, 255, 255), 2)
    
        dist_left = car_position_x - avg_red
        dist_right = avg_blue - car_position_x

        return avg_red, avg_blue, center_x, dist_left, dist_right
    
    return None, None, None, None, None


# Function to draw small green dots along red and blue lines: 
def draw_dots_from_mask(mask, frame, color=(0, 255, 0), y_focus_area = 0):
    red_blue_edges, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for edge in red_blue_edges:
        if cv2.contourArea(edge) > 100:  #100 - ignore noise, possible to try 50 or 30 and see what will happen
            for point in edge:
                x, y = point[0]
                if y >= y_focus_area:
                    cv2.circle(frame, (x, y), 2, color, -1)

# this function will call 2 above functions
def process_frame(frame):
    # 1. Convert to HSV:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. Create masks:
    mask_red = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)
    mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)

    # 3. Define the Y-coordinate focus area (about 70% from the bottom up will be a focus area)
    height = frame.shape[0]
    y_focus_area = int(height * 0.30)

    # Draw green dots on detected red/blue contours
    draw_dots_from_mask(mask_red, frame, (0, 255, 0), y_focus_area)
    draw_dots_from_mask(mask_blue, frame, (0, 255, 0), y_focus_area)

    # 4. Line detection:
    # Draw the center line (the line your car will follow) based on the detected points
    # and compute distance:
    red_x, blue_x, center_x, dist_left, dist_right = compute_center_line_and_distances(
    mask_red, mask_blue, frame, y_focus_area)

    
    # 5. Decide based on center position:
    if center_x is not None:
            
        print(f"Distance to Red: {dist_left}, Distance to Blue: {dist_right}, Center X: {center_x}")
    
        if abs(dist_left - dist_right) < 10:
            return 'F'
        elif dist_left > dist_right:
            return 'L'  # Closer to blue → steer left
            
        else:
            return 'R'  # Closer to red → steer right
    else:
        return 'F'      # No lines found but we will send Forward command
    

# Yolo5 object detection function:
# Load the model (it will load the model once):
model = torch.hub.load('ultralytics/yolov5', # main model
                       'custom', # custom model
                       path='best.pt', # out model
                       skip_validation=True) # no checking

# print(model)

def detect_objects_with_yolo(model, frame):
    results = model(frame)
    results.render()  # automatically draws boxes on frame

    detections = results.pandas().xyxy[0]
    #print(detections) -->> it will print this : Columns: [xmin, ymin, xmax, ymax, confidence, class, name]
    labels_found = []

    for _, obj in detections.iterrows():
        label = obj['name']
        conf = obj['confidence']
        labels_found.append(f"{label} ({conf:.0%})")

        
    detect_object_text = "Detected: " + ", ".join(labels_found) if labels_found else "No objects"
    return detect_object_text

# weather condition function:
def load_weather(): 
    from joblib import load
    import urllib.request, json
  

    # load the model that was said in the training code
    model = load("weathercontrolmodel.joblib")

    # copy the labels from the training code
    # make sure the order is identical!
    labels = ["Ice", "Normal", "Rain", "Snow"]

    data = None

    # download the current weather data from API
    with urllib.request.urlopen("https://edu.frostbit.fi/api/road_weather/2025/") as url:
        data = json.load(url)


    weather = ""   
    # if data was successfully downloaded, make a prediction with our model
    if data != None:
        
        # let's use this in the model, prediction
        tester_row = pd.DataFrame([data])

        result = labels[model.predict(tester_row)[0]]    

        # this can be implemented as you wish in your own car system
        # "decision logic" based on the model's prediction
        if result == "Normal":
            weather = "Weather: Normal"
        elif result == "Rain":
            weather = "Weather: Rainy"
        elif result == "Snow":
            weather = "Weather: Snowy"
        elif result == "Ice":
            weather = "Weather: Icy"
        
    return weather

# MAIN LOOP IS HERE -->>>

frame_count = 0
while True:
    ret, frame = camera.read()
    if not ret or frame is None:
        print("Failed to read frame")
        break
    
    command = process_frame(frame)

    # Run YOLOv5 detection on the same frame
    detect_object_text = detect_objects_with_yolo(model, frame)

    
    # show the command text on the video:    
    cv2.putText(
    frame,                      
    f"Command: {command}",      
    (30, 50),                   # Bottom-left corner of the text
    cv2.FONT_HERSHEY_SIMPLEX,  # Font
    1.2,                        # Font scale
    (0, 255, 255),              # Text color (yellow)
    3,                          # Thickness
    cv2.LINE_AA                # Line type
    )

    # Run weather prediction function:
    weather_text = load_weather()

    # show weather on video:
    cv2.putText(
        frame, weather_text, (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
    )

    # Show processed result
    cv2.imshow("Line Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()  