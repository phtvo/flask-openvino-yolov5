from .base import yolo 
from .modelbooks import ModelBooks

def create_model(name, path2xml,  nclasses, classes=None, device='CPU', 
                cpu_extension=None, conf_thresh=0.5, iou_thresh=0.5) -> yolo:

    assert name in ModelBooks, "Unsupported type {}. Please supported type {}".format(name, list(ModelBooks.keys()))

    _yolo = yolo(path2xml, nclasses, ModelBooks[name], classes=classes, device=device, cpu_extension=cpu_extension,
                            conf_thresh=conf_thresh, iou_thresh=iou_thresh)

    return _yolo 