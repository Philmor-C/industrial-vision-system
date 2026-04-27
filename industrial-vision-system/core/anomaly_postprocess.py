import cv2
import numpy as np

def heatmap_to_boxes(heatmap, threshold=0.6):

    binary = (heatmap > threshold).astype(np.uint8)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        score = float(np.mean(heatmap[y:y+h, x:x+w]))
        boxes.append((x, y, w, h, score))

    return boxes