from ultralytics import YOLO

model = YOLO("models/yolo.pt")

def run_yolo(image):
    results = model(image)[0]

    boxes = []

    if results.boxes is not None:
        for b in results.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            conf = float(b.conf[0])
            cls = int(b.cls[0])

            boxes.append((x1, y1, x2, y2, cls, conf))

    return boxes