CLASS_NAMES = (
    "person",
    "boat",
    "animal",
    "seat",
    "sign",
    "bicycle",
    "car",
    "ball",
    "light",
    "garbage_can",
    "uav",
    "tricycle",
)

NUM_CLASSES = len(CLASS_NAMES)
MODALITIES = ("visible", "infrared", "depth")

# YOLOv5 P3/P4/P5 anchors, in pixels at the training image scale.
DEFAULT_ANCHORS = (
    ((10, 13), (16, 30), (33, 23)),
    ((30, 61), (62, 45), (59, 119)),
    ((116, 90), (156, 198), (373, 326)),
)

