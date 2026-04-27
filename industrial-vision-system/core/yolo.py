try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError(
        "YOLO failed to import. This is usually a missing libGL issue."
    ) from e
from huggingface_hub import hf_hub_download


def load_yolo():
    # download model from HuggingFace
    path = hf_hub_download("Filiyo/yolo", "yolo.pt")
  
    model = YOLO(path)

    return model


def run_yolo(image, model):

    results = model(image)

    boxes = []

    for r in results:

        if r.boxes is None:
            continue

        for b in r.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = b
            boxes.append((float(x1), float(y1), float(x2), float(y2)))

    return boxes
