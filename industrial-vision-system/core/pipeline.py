from core.yolo import run_yolo
from core.patchcore import run_patchcore
from core.fusion import fuse
from core.anomaly_postprocess import heatmap_to_boxes
from core.cv_measure import measure

import torchvision.transforms as T

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor()
])


def run_pipeline(image, yolo_model, memory_bank, threshold):

    # YOLO
    yolo_boxes = run_yolo(image, yolo_model)

    # PatchCore
    img_tensor = transform(image).unsqueeze(0)

    heatmap, score = run_patchcore(img_tensor, memory_bank)

    anomaly_boxes = heatmap_to_boxes(heatmap, threshold)

    # SCALE FIX (224 → original)
    w, h = image.size
    scale_x = w / 224
    scale_y = h / 224

    anomaly_boxes = [
        (
            int(x1 * scale_x),
            int(y1 * scale_y),
            int(x2 * scale_x),
            int(y2 * scale_y),
        )
        for (x1,y1,x2,y2) in anomaly_boxes
    ]

    # Fusion
    matched, unknown = fuse(yolo_boxes, anomaly_boxes)

    measurement = measure(image)

    return {
        "yolo": yolo_boxes,
        "anomaly": anomaly_boxes,
        "matched": matched,
        "unknown": unknown,
        "score": score,
        "measurement": measurement
    }
