import numpy as np 
import cv2 
from numba import jit, njit

def letterbox(img, size=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    w, h = size

    # Scale ratio (new / old)
    r = min(h / shape[0], w / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = w - new_unpad[0], h - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (w, h)
        ratio = w / shape[1], h / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    top2, bottom2, left2, right2 = 0, 0, 0, 0
    if img.shape[0] != h:
        top2 = (h - img.shape[0])//2
        bottom2 = top2
        img = cv2.copyMakeBorder(img, top2, bottom2, left2, right2, cv2.BORDER_CONSTANT, value=color)  # add border
    elif img.shape[1] != w:
        left2 = (w - img.shape[1])//2
        right2 = left2
        img = cv2.copyMakeBorder(img, top2, bottom2, left2, right2, cv2.BORDER_CONSTANT, value=color)  # add border
    return img


class YoloParams:
    # ------------------------------------------- Extracting layer parameters ------------------------------------------
    # Magic numbers are copied from yolo samples
    def __init__(self, param):
        self.num_anchors = 3 if 'num_anchors' not in param else int(param['num_anchors'])
        self.coords = 4 if 'coords' not in param else int(param['coords'])
        self.classes = 80 if 'classes' not in param else int(param['classes'])
        #self.grid = grid
        self._anchors = [19,27,  44,40,  38,94 , 96,68,  86,152,  180,137  , 140,301,  303,264,  238,542 , 436,615,  739,380,  925,792 ] if 'anchors' not in param else param['anchors']

        n = len(self._anchors) // (self.num_anchors * 2)
        anchors = [ [self._anchors[i], self._anchors[i+1]] for i in range(0, len(self._anchors), 2)]
        self.anchors = np.split(np.array(anchors), n)

parse_idx = {
    8: 0, 16: 1, 32: 2, 64: 3
}

def parse_scale(resize, blob_size):
    #print(int(resize/blob_size))

    return parse_idx[int(resize/blob_size)]

@njit(parallel=True)
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

#@jit(parallel=True)
def parse_yolo_detection(blob, resized_image_shape, original_im_shape, 
                    anchors, num_anchors, threshold):
    """Convert Detection Conv to [xmin, ymin, xmax, ymax, conf, class]

    Args:
        blob (np.ndarray): shape 1x((4 + 1 + n_class) * 3)x grid x grid
        resized_image_shape (tuple): resize image (W, H)
        original_im_shape (tuple): originial image size (W, H)
        anchors (list): list of this layer's anchors
        num_anchors (itn): number of anchors of this layer
        threshold (float): confidence threshold

    Returns:
        [np.ndarray]: tensor shape (number of sample has confidence >= threshold , 4 + 1 + 1) [[xmin, ymin, xmax, ymax, conf, class]]
    """
    # ------------------------------------------ Validating output parameters ------------------------------------------    
    batch, channels, grid, grid2 = blob.shape
    #tic = time()
    predictions = sigmoid(blob)#1.0/(1.0+np.exp(-blob))  # sigmoid
    #predictions = 1.0/(1.0+np.exp(-blob))  # sigmoid
    #print(f'Sigmoid time: {round((time() - tic) *1000, 5)}ms')

    # ------------------------------------------ Extracting layer parameters -------------------------------------------
    orig_im_w, orig_im_h = original_im_shape
    resized_image_w, resized_image_h = resized_image_shape
    bbox_size = int(channels/num_anchors) #4+1+num_classes
    img_grid_ratio = resized_image_w / grid
    #print(grid)
    #print(anchors)
    #anchors  = anchors[parse_scale(resized_image_w, grid)]
    #tic = time()
    _anchors = np.repeat(anchors, [grid**2, grid**2, grid**2], axis=0)
    _anchors = _anchors.reshape(1, num_anchors, grid, grid, 2).astype('float16') #/ img_grid_ratio
    #print(f'Anchors time: {round((time() - tic) *1000, 5)}ms')
    # ------------------------------------------ Extracting Blob data -------------------------------------------
    # blob size : n x c x grid x grid
    # c: x y w h conf clss1 clss2 .. clssn
    #tic = time()
    mesh = np.meshgrid(np.arange(grid), np.arange(grid) )
    mesh = np.array((mesh[0], mesh[1]))
    mesh = np.stack(mesh, axis=2).reshape(1,1,grid,grid,2).astype('float16')
    predictions = predictions.reshape(batch, num_anchors , bbox_size, grid, grid).transpose(0, 1, 3, 4, 2).astype('float16')
    #print(f'Reshape time: {round((time() - tic) *1000, 5)}ms')
    #tic = time()
    confidence = predictions[..., 4]
    confidence[ confidence < threshold] = 0
    
    xy = (2 * predictions[..., 0:2] - 0.5 + mesh) *float(img_grid_ratio)
    wh = (2 * predictions[..., 2:4] )**2 * _anchors 
    xy = xy[confidence != 0]
    wh = wh[confidence != 0]
    clss = np.argmax(predictions[..., 5:][confidence != 0], axis=1)
    # ---------------------------------------- Scale Input -----------------------------------------#
    diff = abs(orig_im_h - orig_im_w) / 2.
    pad = (diff, 0) if orig_im_h > orig_im_w else (0, diff)
    pad = np.asarray(pad).astype('float16')
    scale = max(orig_im_w, orig_im_h) / resized_image_h
    xy = (xy * float(scale)) - pad
    wh = wh*float(scale)
    xymin = np.maximum(0, xy - wh/2)
    xymax = np.minimum([orig_im_w, orig_im_h], xymin + wh)
    
    confidence = confidence[confidence != 0]
    return np.concatenate((xymin, xymax, confidence.reshape(-1, batch), clss.reshape(-1, batch)), axis=1)

def bbox_iou_numpy(box1, box2, x1y1x2y2=True):
    """Computes IoU between bounding boxes.
    Parameters
    ----------
    box1 : ndarray
        (N, 4) shaped array with bboxes
    box2 : ndarray
        (M, 4) shaped array with bboxes
    Returns
    -------
    : ndarray
        (N, M) shaped array with IoUs
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,
                                          0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,
                                          0], box2[:, 1], box2[:, 2], box2[:, 3]
    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2).clip(min=0.)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2).clip(min=0.)
    # Intersection area
    inter_area = (inter_rect_x2 - inter_rect_x1 + 1).clip(min=0.) * (inter_rect_y2 - inter_rect_y1 + 1).clip(min=0.)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    # you may get shit result if type is not float
    iou = inter_area.astype('float') / (np.abs(b1_area + b2_area - inter_area + 1e-8)).astype('float')
    return iou


def nms(predictions, nclasses, conf_thresh=None, iou_thresh=0.65):
    """
    Predictions: xmin ymin xmax ymax conf class
    """
    if not conf_thresh is None:
        conf = predictions[..., 4]
        conf_mask = conf[conf >= conf_thresh].squeeze()
        predictions = predictions[conf_mask]
    #class_conf=  predictions[:, -1]
    # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
    #predictions = np.concatenate(
    #        (predictions[:, :5],  class_conf,predictions[:, -1].reshape(-1, 1) ), 1)
    unique_labels = np.unique(predictions[:, -1])
    result = None
    for label in unique_labels:
        detected = predictions[predictions[:,-1] == label]
        ind_conf = np.argsort(detected[:, 4], axis=0)[::-1]
        detected = detected[ind_conf]
        max_detections = []
        while detected.shape[0]:
            max_detections.append(np.expand_dims(detected[0], 0))
            if len(detected) == 1:
                break
            # Get the IOUs for all boxes with lower confidence
            ious = bbox_iou_numpy(max_detections[-1][...,:4], detected[1:][...,:4])
            # Remove detections with IoU >= NMS threshold
            detected = detected[1:][ious < iou_thresh]
        max_detections = np.concatenate(max_detections)
        result = np.concatenate([result, max_detections]) if not result is None else max_detections
    return result

def _nms(predictions, iou_thresh, **kwargs):
    boxes = predictions
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes	
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > iou_thresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

def pad_img(img, img_shape=(640,640)):
    h, w = img.shape[:2] 
    #
    dim_diff = np.abs(h - w)
    # Upper (left) and lower (right) padding
    pad1, pad2 = dim_diff // 2, dim_diff // 2
    # Determine padding
    pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else (
        (0, 0), (pad1, pad2), (0, 0))
    # Add padding
    input_img = np.pad(img.copy(), pad, 'constant', constant_values=0)
    
    #padded_h, padded_w, _ = input_img.shape
    return cv2.resize(input_img, img_shape)