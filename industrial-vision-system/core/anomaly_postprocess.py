import numpy as np
import cv2

def heatmap_to_boxes(heatmap, threshold):

    heatmap = (heatmap * 255).astype("uint8")

    _, binary = cv2.threshold(
        heatmap,
        int(threshold * 255),
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, x+w, y+h))

    return boxes
