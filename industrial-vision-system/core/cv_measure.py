import cv2

def measure(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)

    x,y,w,h = cv2.boundingRect(c)
    area = cv2.contourArea(c)

    return {"bbox": (x,y,w,h), "area": area}