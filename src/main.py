import os
import cv2
import uuid
from fastapi import FastAPI, File, UploadFile, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.processing import process_frame
import numpy as np

app = FastAPI(title="Plant Recognition & Detection API")

# Папки для статики и загрузок
os.makedirs("src/static", exist_ok=True)
os.makedirs("src/templates", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

app.mount("/static", StaticFiles(directory="src/static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
templates = Jinja2Templates(directory="src/templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Инференс
    processed_img = process_frame(image)
    
    # Сохранение результата
    filename = f"{uuid.uuid4()}.jpg"
    filepath = os.path.join("uploads", filename)
    cv2.imwrite(filepath, processed_img)
    
    return {"url": f"/uploads/{filename}"}

@app.post("/predict/video")
async def predict_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # Сохраняем исходное видео
    input_filename = f"in_{uuid.uuid4()}.mp4"
    input_filepath = os.path.join("uploads", input_filename)
    
    with open(input_filepath, "wb") as buffer:
        buffer.write(await file.read())
        
    output_filename = f"out_{uuid.uuid4()}.mp4"
    output_filepath = os.path.join("uploads", output_filename)
    
    # Обработка видео: читаем кадры, отправляем в модель, пишем новое видео
    cap = cv2.VideoCapture(input_filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filepath, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame = process_frame(frame)
        out.write(processed_frame)
        
    cap.release()
    out.release()
    
    # Возвращаем ссылку на обработанное видео
    return {"url": f"/uploads/{output_filename}"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
