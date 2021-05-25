import os 
import torch 
import pytesseract
import numpy as np 
import config

from PIL import ImageFont, ImageDraw, Image
fontpath ='./data/hgrpp1.ttc' 
font = ImageFont.truetype(fontpath, 25) 

img2text = lambda rgb : pytesseract.image_to_string(rgb, 'jpn', config=r' --psm 3 --oem 1')

from copy import deepcopy

def dict_of_dicts_merge(x, y):
    z = {}
    overlapping_keys = x.keys() & y.keys()
    for key in overlapping_keys:
        z[key] = dict_of_dicts_merge(x[key], y[key])
    for key in x.keys() - overlapping_keys:
        z[key] = deepcopy(x[key])
    for key in y.keys() - overlapping_keys:
        z[key] = deepcopy(y[key])
    return z

class ai:

    def __init__(self, weightpath, img_size= 960) -> None:
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path= weightpath, force_reload=False)
        self.model.conf = config.CONFIDENCE_THRESHOLD  # confidence threshold (0-1)
        self.model.iou = config.IOU_THRESHOLD  # NMS IoU threshold (0-1)
        self.text_items = ['row', 'submit_button']
        self.img_size = img_size
        self.calibx = 20
        self.caliby = 5
    def detect_obj(self, image:np.ndarray):
        api = {}
        h,w = image.shape[:-1]
        result = self.model(image, size=self.img_size) # <------------------------ reduce size for faster
        for row in result.pandas().xyxy[0].iterrows():
            name = row[1]['name']
            xmin = max(0, int(row[1]['xmin'] - self.calibx))
            xmax = min(int(row[1]['xmax'] + self.calibx), w)
            ymax = min(int(row[1]['ymax'] + self.caliby), h)
            ymin = max(0, int(row[1]['ymin'] - self.caliby))
            if not name in api:
                api.update({name : [(xmin, ymin, xmax, ymax) ] })   
            else:
                api[name].append((xmin, ymin, xmax, ymax))
        return api 

    def detect_deeply(self, image):
        """
        Only form
        """
        result_phase1 = self.detect_obj(image)
        apis = []
        if 'webform' in result_phase1:
            for form in result_phase1['webform']:
                xmin, ymin, xmax, ymax = form
                img = image[ymin:ymax, xmin:xmax]
                api = self.detect_obj(img)
                api = dict((k, [ [item[0] + xmin, item[1] + ymin, item[2] + xmin, item[3] + ymin] for item in v]) for k, v in api.items())
                apis.append(api)
            result = apis[0]  if len(apis) < 2 else dict_of_dicts_merge(apis[0], apis[1])
            result_phase1.update(result)
        
        return result_phase1


    def process_form(self, image, draw=True, deeply=False):
        if draw:
            tmp_img = image.copy()
            tmp_img = Image.fromarray(tmp_img)
            draw = ImageDraw.Draw(tmp_img)
        if not deeply:
            result = self.detect_obj(image)
        else:
            result = self.detect_deeply(image)
        apis = {}
        for item, value in result.items():
            apis.update({item : []})
            for (xmin, ymin, xmax, ymax ) in value:
                if item in self.text_items:
                    text = img2text(image[ymin:ymax, xmin:xmax])
                    text = text.replace('\n', '').replace('\x0c', '').replace(' ', '')
                    if draw:
                        draw.text(((xmax + xmin * 0.7)//2 ,  max(0, ymin + 30)), text, font = font , fill = (0, 12, 223, 0) )
                    apis[item].append([text, (xmin, ymin, xmax, ymax) ])
                else:
                    apis[item].append(['', (xmin, ymin, xmax, ymax) ])

                if draw:
                    draw.text(( (xmax - 50) , ymax - 40 ), item, font = font , fill = (15, 0, 43, 0) )
                    draw.rectangle(((xmin, ymin ),(xmax, ymax)), outline=(155, 233, 23), fill=None, width=2)
        #apis.update()
        return (apis, None) if not draw else (apis, tmp_img)
    
