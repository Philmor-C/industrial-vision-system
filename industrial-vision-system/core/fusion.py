def iou(a, b):
    x1,y1,x2,y2 = a
    x3,y3,x4,y4 = b

    xa = max(x1, x3)
    ya = max(y1, y3)
    xb = min(x2, x4)
    yb = min(y2, y4)

    inter = max(0, xb-xa) * max(0, yb-ya)

    area1 = (x2-x1)*(y2-y1)
    area2 = (x4-x3)*(y4-y3)

    return inter / (area1 + area2 - inter + 1e-6)


def fuse(yolo_boxes, anomaly_boxes, thresh=0.3):

    matched = []
    unknown = []

    for ab in anomaly_boxes:
        found = False

        for yb in yolo_boxes:
            y_box = (yb[0], yb[1], yb[2], yb[3])

            if iou(ab, y_box) > thresh:
                matched.append((yb, ab))
                found = True
                break

        if not found:
            unknown.append(ab)

    return matched, unknown