from core.yolo import run_yolo
from core.patchcore import run_patchcore
from core.anomaly_postprocess import heatmap_to_boxes
from core.fusion import fuse
from core.cv_measure import measure


def run_pipeline(image, yolo_model, backbone, memory_bank):

    try:
        yolo_boxes = run_yolo(image, yolo_model)
    except:
        yolo_boxes = []

    try:
        heatmap, score = run_patchcore(image, backbone, memory_bank)
        anomaly_boxes = heatmap_to_boxes(heatmap)
    except:
        heatmap, score = None, 0
        anomaly_boxes = []

    matched, unknown = fuse(yolo_boxes, anomaly_boxes)

    measurement = measure(image)

    return {
        "yolo": yolo_boxes,
        "anomaly": anomaly_boxes,
        "unknown": unknown,
        "matched": matched,
        "score": score,
        "heatmap": heatmap,
        "measurement": measurement
    }
