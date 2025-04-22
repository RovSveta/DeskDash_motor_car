from fastapi import FastAPI, Request, Depends, Response, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from app.auth import authenticate
from .camera import camera_stream
from pydantic import BaseModel
import serial

ser = serial.Serial('/dev/ttyUSB0', 115200)

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

class MoveCommand(BaseModel):
    command: str

@app.post("/move")
async def move(cmd: MoveCommand):
    print("Move:", cmd.command)
    # Here you'd send the command to your car (e.g., via GPIO or serial)
    
    if cmd.command in ['f', 'b', 'l', 'r', 's']:
        ser.write(cmd.command.encode())  # Send command over serial
        return {"status": "ok", "sent": cmd.command}
    else:
        return {"status": "error", "message": "Invalid command"}
    

@app.get("/", response_class=HTMLResponse)
def home(request: Request, creds=Depends(authenticate)):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video")
def video_feed():
    def generate():
        while True:
            frame = camera_stream.get_frame()
            if frame:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/start")
def start_stream():
    camera_stream.set_running(True)
    return {"status": "started"}

@app.post("/stop")
def stop_stream():
    camera_stream.set_running(False)
    return {"status": "stopped"}
