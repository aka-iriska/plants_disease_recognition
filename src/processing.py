import cv2
import numpy as np
import onnxruntime as ort
from typing import Tuple

# Загрузка ONNX модели (лениво)
_session = None

def get_session():
    global _session
    if _session is None:
        try:
            # Пытаемся загрузить модель. Если ее нет, будем мокать результат для демо
            _session = ort.InferenceSession("weights/plant_model.onnx", providers=['CPUExecutionProvider'])
        except Exception as e:
            print(f"ONNX Model not found or failed to load: {e}. Using mock predictions.")
            _session = "MOCK"
    return _session

def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = (640, 640)):
    # Изменение размера с сохранением пропорций (padding)
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    
    resized = cv2.resize(image, (nw, nh))
    
    # Padding
    pad_h = target_size[0] - nh
    pad_w = target_size[1] - nw
    
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    # HWC to CHW, BGR to RGB, нормализация
    input_data = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    input_data = input_data.transpose((2, 0, 1)).astype(np.float32)
    input_data /= 255.0
    input_data = np.expand_dims(input_data, axis=0) # Batch dimension
    
    return input_data, (scale, pad_w // 2, pad_h // 2)

def draw_boxes(image: np.ndarray, boxes, scores, class_ids, labels: list):
    for box, score, cls_id in zip(boxes, scores, class_ids):
        if score < 0.5:
            continue
        x1, y1, x2, y2 = map(int, box)
        label = f"{labels[cls_id] if cls_id < len(labels) else 'Plant'} {score:.2f}"
        
        # Рисуем бокс
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Рисуем подложку для текста
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1 - h - 5), (x1 + w, y1), (0, 255, 0), -1)
        # Рисуем текст
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return image

def process_frame(frame: np.ndarray) -> np.ndarray:
    session = get_session()
    
    if session == "MOCK":
        # Рисуем фейковый бокс по центру, если модели еще нет
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (255, 0, 0), 3)
        cv2.putText(frame, "Mock Plant 0.99", (w//4, h//4 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        return frame
        
    input_data, (scale, pad_x, pad_y) = preprocess_image(frame)
    input_name = session.get_inputs()[0].name
    
    # Инференс ONNX
    outputs = session.run(None, {input_name: input_data})
    
    # Парсинг результатов зависит от архитектуры (YOLOv8, Faster R-CNN и т.д.)
    # Здесь пример для гипотетического выхода: [boxes, scores, class_ids]
    # Тебе нужно будет адаптировать это под свою модель!
    try:
        boxes, scores, class_ids = outputs[0], outputs[1], outputs[2]
        
        # Пересчет координат обратно (отмена padding и scale)
        # boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        # boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
        
        labels = ["Healthy", "Diseased"] # Пример классов
        frame = draw_boxes(frame, boxes, scores, class_ids, labels)
    except Exception as e:
        # Если выходы не совпадают, просто возвращаем кадр для отказоустойчивости
        pass
        
    return frame
