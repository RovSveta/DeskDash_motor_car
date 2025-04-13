def load_weather(): 
    import numpy as np
    from joblib import load
    import urllib.request, json
    import pandas as pd

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

import cv2

# Get weather info
weather_text = load_weather()

# Initialize camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw weather text on video frame
    cv2.putText(frame, weather_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('DeskDash Video Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


