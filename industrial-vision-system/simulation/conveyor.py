class ConveyorItem:
    def __init__(self, idx, image):
        self.idx = idx
        self.image = image
        self.x = 0
        self.result = None


# =========================
# encoder movement simulation
# =========================
def encoder_move(items, speed=5):

    for item in items:
        item.x += speed


# =========================
# trigger logic (inspection zone)
# =========================
def check_trigger(item, zone=250):

    return item.x >= zone and item.result is None