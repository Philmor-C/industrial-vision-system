from core.yolo import run_yolo
from core.patchcore import run_patchcore
from core.fusion import fuse
from core.anomaly_postprocess import heatmap_to_boxes
from core.cv_measure import measure

def run_pipeline(image, yolo_model, backbone, memory_bank):

    yolo_boxes = run_yolo(image, yolo_model)

    heatmap, score = run_patchcore(image, backbone, memory_bank)

    anomaly_boxes = heatmap_to_boxes(heatmap)

    matched, unknown = fuse(yolo_boxes, anomaly_boxes)

    measurement = measure(image)

    return {
        "yolo": yolo_boxes,
        "anomaly": anomaly_boxes,
        "matched": matched,
        "unknown": unknown,
        "score": score,
        "heatmap": heatmap,
        "measurement": measurement
    }
