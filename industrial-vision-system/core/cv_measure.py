def measure(image):

    w, h = image.size

    return {
        "width_mm": w * 0.1,
        "height_mm": h * 0.1,
        "area_mm2": (w * h) * 0.01
    }
