import torch
from huggingface_hub import hf_hub_download

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_yolo():
    path = hf_hub_download("Filiyo/yolo", "yolo.pt")
    model = torch.load(path, map_location=DEVICE)
    model.eval()
    return model


def run_yolo(image, model):
    # image: PIL or tensor
    results = model(image)

    boxes = []

    try:
        for r in results:
            for b in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = b
                boxes.append((x1, y1, x2, y2))
    except:
        pass

    return boxes
