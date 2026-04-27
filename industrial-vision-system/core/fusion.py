def iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2-x1) * max(0, y2-y1)

    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])

    return inter / (a1 + a2 - inter + 1e-6)


def fuse(yolo_boxes, anomaly_boxes, thr=0.2):

    matched = []
    unknown = []

    for a in anomaly_boxes:

        found = False

        for y in yolo_boxes:
            if iou(a, y) > thr:
                matched.append((a, y))
                found = True
                break

        if not found:
            unknown.append(a)

    return matched, unknown
