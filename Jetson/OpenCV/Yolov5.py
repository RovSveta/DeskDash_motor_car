import torch
import cv2  

# Yolo5 object detection function:
# Append posix to the system path
### Этот фикс только для Windows, чтобы избежать ошибки с путями. На линуксе не нужно.
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
###
 
# Load the YOLOv5 model from the specified path
model = torch.hub.load('ultralytics/yolov5', # Основная модель
                       'custom', # Указание на кастомную модель
                       path='best.pt', # Путь к файлу модели
                       skip_validation=True) # Пропуск проверки
print(model)
 
 
# Create camera reader
camera = cv2.VideoCapture(0)
 
# Read the camera, plot the results and display the image
while True:
    ret, frame = camera.read()
    if not ret:
        break
 
    # Perform inference on the frame
    results = model(frame)
 
    # Plot the results on the frame
    results.render()
 
    # Display the image with detections
    cv2.imshow('YOLOv5 Detection', frame)
 
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break