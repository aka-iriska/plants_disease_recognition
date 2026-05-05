from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import torch
import torchvision.transforms as T
import io

app = FastAPI(title="Plant Recognition API")

# 1. Загрузка модели (путь к весам из Colab)
MODEL_PATH = "weights/plant_model.pth"
model = None

@app.on_event("startup")
def load_model():
    global model
    # Здесь должна быть твоя архитектура, например: model = MyCNN()
    # model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    # model.eval()
    print("Модель успешно загружена!")

# Препроцессинг картинки (должен совпадать с тем, что был в Colab)
transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")

    # Читаем картинку
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")

    # Подготовка и инференс
    # input_tensor = transforms(image).unsqueeze(0)
    # with torch.no_grad():
    #     output = model(input_tensor)
    #     predicted_class = output.argmax(1).item()

    return {"label": "Daisy (placeholder)", "confidence": 0.98}
