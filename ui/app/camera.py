import cv2
import torch

class CameraStream:
    def __init__(self, cam_index=0, model_path='best.pt'):
        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

        # Initialize camera
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise Exception("❌ Error: Could not open USB camera.")

        self.running = True

    def get_frame(self):
        if not self.running:
            return None

        ret, frame = self.cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            return None

        # Run YOLO inference
        results = self.model(frame)
        results.render()

        # Get the rendered frame
        annotated_frame = results.ims[0]  # RGB numpy array
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

        # Optionally print detections
        detections = results.pandas().xyxy[0]
        for name in detections['name'].unique():
            print(f"Detected: {name}")

        # Encode to JPEG for streaming
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        return buffer.tobytes()

    def set_running(self, status: bool):
        self.running = status

    def release(self):
        self.cap.release()


# Initialize camera stream
camera_stream = CameraStream(cam_index=0, model_path='best.pt')
