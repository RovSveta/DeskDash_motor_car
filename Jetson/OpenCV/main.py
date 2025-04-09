import cv2
import numpy as np
import pandas as pd
import os

# Open USB camera:
#camera = cv2.VideoCapture(0)
camera = cv2.VideoCapture("VID_20250401_122228.mp4")

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

def compute_center_line_and_distances(mask_red, mask_blue, frame, y_focus_area):
    red_x = []
    blue_x = []

    red_line_edges, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_line_edges, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for edge in red_line_edges:
        if cv2.contourArea(edge) > 100:
            for point in edge:
                x, y = point[0]
                if y >= y_focus_area:
                    red_x.append(x)

    for edge in blue_line_edges:
        if cv2.contourArea(edge) > 100:
            for point in edge:
                x, y = point[0]
                if y >= y_focus_area:
                    blue_x.append(x)

    if red_x and blue_x:
        avg_red = int(np.mean(red_x))
        avg_blue = int(np.mean(blue_x))
        center_x = (avg_red + avg_blue) // 2
        
        # this is a car position, middle of x axe
        width = frame.shape[:2]
        car_position_x = width // 2
        cv2.line(frame, (center_x, 0), (center_x, frame.shape[0]), (0, 255, 255), 2)
    
        dist_left = car_position_x - avg_red
        dist_right = avg_blue - car_position_x

        return avg_red, avg_blue, center_x, dist_left, dist_right
    
    return None, None, None, None, None


# Function to draw small green dots along red and blue lines: 
def draw_dots_from_mask(mask, frame, color=(0, 255, 0), y_focus_area = 0):
    red_blue_edges, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for edge in red_blue_edges:
        if cv2.contourArea(edge) > 100:  #100 - ignore noise
            for point in edge:
                x, y = point[0]
                if y >= y_focus_area:
                    cv2.circle(frame, (x, y), 2, color, -1)


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
            return 'Forward'
        elif dist_left > dist_right:
            return 'L'  # Closer to blue → steer left
            
        else:
            return 'R'  # Closer to red → steer right
    else:
        return 'No lines'      # No lines found

# MAIN LOOP IS HERE -->>>

frame_count = 0
while True:
    ret, frame = camera.read()
    if not ret or frame is None:
        print("Failed to read frame")
        break
    
    command = process_frame(frame)
    
    # show the command text on the video
    
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


    # Show processed result
    cv2.imshow("Line Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()