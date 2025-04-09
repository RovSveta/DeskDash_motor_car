import cv2
import numpy as np
import pandas as pd
import os

# folder paths:
input_folder = "C:/Storage/Studies/Lapland_AMK/4_semester/Robotics_project/OpenCV/Frames"  
output_folder = "C:/Storage/Studies/Lapland_AMK/4_semester/Robotics_project/OpenCV/ProcessedFrames"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

results = []

# Adjusted HSV Ranges
# Red
red_lower1 = np.array([0, 70, 70])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([160, 70, 70])
red_upper2 = np.array([179, 255, 255])

# Blue 
blue_lower = np.array([85, 40, 40])
blue_upper = np.array([135, 255, 255])


# Function to draw small green dots along red and blue lines: 
def draw_dots_from_mask(mask, image, color=(0, 255, 0)):
    red_blue_edges, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for edge in red_blue_edges:
        if cv2.contourArea(edge) > 100:  #100 - ignore noise
            for point in edge:
                x, y = point[0]
                if y >= y_focus_area:
                    cv2.circle(image, (x, y), 2, color, -1)

def compute_center_line_and_distances(mask_red, mask_blue, image, y_focus_area):
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
        cv2.line(image, (center_x, 0), (center_x, image.shape[0]), (0, 255, 255), 2)
    
        dist_left = abs(center_x - avg_red)
        dist_right = abs(center_x - avg_blue)

        return avg_red, avg_blue, center_x, dist_left, dist_right
    return None, None, None, None, None



frame_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith((".jpg"))])
for file in frame_files:
    image_path = os.path.join(input_folder, file)
    image = cv2.imread(image_path)
    if image is None:
        continue
    
    # Convert to HSV for color-based detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_red = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)
    mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)
    
    # Define the Y-coordinate focus area (about 70% from the bottom up will be a focus area)
    height = image.shape[0]
    y_focus_area = int(height * 0.30)
    
    
    # Draw green dots on the red and blue masks in the bottom 25%
    draw_dots_from_mask(mask_red, image, (0, 255, 0))
    draw_dots_from_mask(mask_blue, image, (0, 255, 0))
    
    # Draw the center line (the line your car will follow) based on the detected points
    # and compute distance:
    red_x, blue_x, center_x, dist_left, dist_right = compute_center_line_and_distances(
    mask_red, mask_blue, image, y_focus_area)

    results.append({
        "Frame": file,
        "Red_X": red_x,
        "Blue_X": blue_x,
        "Center_X": center_x,
        "Dist_Left": dist_left,
        "Dist_Right": dist_right
    })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_folder, "lane_positions.csv"), index=False)
 
       # Save the processed frame to the output folder
    out_path = os.path.join(output_folder, file)
    cv2.imwrite(out_path, image)




cv2.destroyAllWindows()



