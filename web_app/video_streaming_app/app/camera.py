import cv2

class CameraStream:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True

    def get_frame(self):
        if not self.running:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        _, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()

    def set_running(self, status: bool):
        self.running = status

    def release(self):
        self.cap.release()

camera_stream = CameraStream()
