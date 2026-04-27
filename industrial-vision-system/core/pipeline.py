import torch
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from core.yolo import run_yolo
from core.patchcore import run_patchcore
from core.anomaly_postprocess import heatmap_to_boxes
from core.fusion import fuse
from core.cv_measure import measure

# =========================
# DEVICE
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# TRANSFORMS (CRITICAL)
# =========================
yolo_transform = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor()
])

patch_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

# =========================
# SCALE FUNCTION (224 → 640)
# =========================
def scale_boxes(boxes, from_size=224, to_size=640):
    scale_x = to_size / from_size
    scale_y = to_size / from_size

    scaled = []
    for x1, y1, x2, y2 in boxes:
        scaled.append((
            x1 * scale_x,
            y1 * scale_y,
            x2 * scale_x,
            y2 * scale_y
        ))
    return scaled

# =========================
# MAIN PIPELINE
# =========================
def run_pipeline(image, yolo_model, patchcore_backbone, memory_bank, threshold):

    """
    image: PIL Image (original high-res)
    """

    # =========================
    # PREPROCESS
    # =========================
    yolo_input = yolo_transform(image).unsqueeze(0).to(DEVICE)
    patch_input = patch_transform(image).unsqueeze(0).to(DEVICE)

    # =========================
    # PARALLEL INFERENCE
    # =========================
    with ThreadPoolExecutor(max_workers=2) as executor:

        yolo_future = executor.submit(run_yolo, yolo_input, yolo_model)

        patch_future = executor.submit(
            run_patchcore,
            patch_input,
            patchcore_backbone,
            memory_bank
        )

        yolo_boxes = yolo_future.result()
        heatmap, score, _ = patch_future.result()

    # =========================
    # PATCHCORE → BOXES (224 SPACE)
    # =========================
    anomaly_boxes_224 = heatmap_to_boxes(heatmap)

    # =========================
    # SCALE TO YOLO SPACE (640)
    # =========================
    anomaly_boxes_640 = scale_boxes(
        anomaly_boxes_224,
        from_size=224,
        to_size=640
    )

    # =========================
    # FUSION (SAME COORD SYSTEM)
    # =========================
    matched, unknown = fuse(
        [(b[0], b[1], b[2], b[3]) for b in yolo_boxes],
        anomaly_boxes_640
    )

    # =========================
    # CLASSICAL CV MEASUREMENTS
    # =========================
    measurement = measure(image)

    # =========================
    # OUTPUT STRUCTURE
    # =========================
    return {
        "yolo_boxes": yolo_boxes,
        "anomaly_boxes": anomaly_boxes_640,
        "matched": matched,
        "unknown": unknown,
        "anomaly_score": score,
        "threshold": threshold,
        "measurement": measurement
    }