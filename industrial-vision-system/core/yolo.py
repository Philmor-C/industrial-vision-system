import torch
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

DEVICE = "cpu"   # IMPORTANT for Streamlit Cloud stability


def load_yolo():

    path = hf_hub_download("Filiyo/yolo", "yolo.pt")

    # FIX: SAFE LOAD for PyTorch 2.6
    model = YOLO(path)

    return model


def run_yolo(image, model):

    results = model(image)

    boxes = []

    for r in results:
        if r.boxes is not None:
            for b in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = b
                boxes.append((x1, y1, x2, y2))

    return boxes
