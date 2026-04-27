class ConveyorItem:
    def __init__(self, idx, image):
        self.idx = idx
        self.image = image
        self.x = idx * 50
        self.result = None


def encoder_move(items, speed=5):
    for i in items:
        i.x += speed


def check_trigger(item, zone):
    return abs(item.x - zone) < 10
