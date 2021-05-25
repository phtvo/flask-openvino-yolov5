from .openvino_yolov5 import create_model
import pytesseract
import numpy as np 
import config

from PIL import ImageFont, ImageDraw, Image
fontpath ='./data/hgrpp1.ttc' 
font = ImageFont.truetype(fontpath, 20) 

if config.EASY_OCR:
    from .ocr import OCR
    ocr = OCR(config.OCR_GPU)
    img2text = ocr.topleft_text
else:
    img2text = lambda rgb : pytesseract.image_to_string(rgb, 'jpn', config=r' --psm 3 --oem 1')

from copy import deepcopy

def dict_of_dicts_merge(x, y):
    z = {}
    #print(type(x), x)
    #print(type(y), y)
    overlapping_keys = x.keys() & y.keys()
    for key in overlapping_keys:
        z[key] = x[key] + y[key]
    for key in x.keys() - overlapping_keys:
        z[key] = deepcopy(x[key])
    for key in y.keys() - overlapping_keys:
        z[key] = deepcopy(y[key])
    return z

class ai:

    def __init__(self, weightpath,  nclasses, classes, 
                    conf= 0.5, iou= 0.5, img_size= 640,
                    calibx=40, caliby = 5
        ) -> None:
        self.model = create_model(
                                    'contactform', weightpath, nclasses=nclasses, 
                                    classes= classes,
                                    conf_thresh=conf, 
                                    iou_thresh=iou
                                    )
        self.text_items = ['row', 'submit_button']
        self.img_size = img_size
        self.calibx = calibx
        self.caliby = caliby
        self.other_color = (255,140,0)  
        self.row_color = [(30,144,255), (255,99,71)]

    def detect_obj(self, image:np.ndarray):
        apis =  self.model.detect_api(image[:,:,::-1])
        h,w = image.shape[:-1]
        for row, items in apis.items():
            tmp = []
            for item in items:
                xmin = max(0, int(item[0] - self.calibx))
                ymin = max(0, int(item[1] - self.caliby))
                xmax = min(int(item[2] + self.calibx), w)
                ymax = min(int(item[3] + self.caliby), h)
                tmp.append((xmin, ymin, xmax, ymax))
            apis[row] = tmp
        return apis 

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
            value.sort(key = lambda x: x[1])
            for i, (xmin, ymin, xmax, ymax ) in enumerate(value):   
                if item in self.text_items:
                    text = img2text(image[ymin:ymax, xmin:xmax])
                    text = text.replace('\n', '').replace('\x0c', '').replace(' ', '')
                    if draw:
                        draw.rectangle(((xmin+10, ymin + 10 ),(xmax-10, ymax -10)), outline=self.row_color[ int(i % 2) ], fill=None, width=2)
                        draw.text(((xmax + xmin * 0.8)//2 ,  ymin -7), text + f"_({item})", font = font , fill = (*( self.row_color[ int(i % 2) ]), 0) )
                        #draw.text(( (xmax - 40) , ymax - 2), item, font = font , fill = (*( self.row_color[ int(i % 2) ]), 0) )
                    apis[item].append([text, (xmin, ymin, xmax, ymax) ])
                else:
                    apis[item].append(['', (xmin, ymin, xmax, ymax) ])
                    if draw:
                        draw.text(( (xmax - 40) , ymin + 10 ), item, font = font , fill = (*self.other_color, 0) )
                        draw.rectangle(((xmin - 10, ymin -10),(xmax + 10, ymax + 10)), outline=self.other_color, fill=None, width=2)

        #apis.update()
        return (apis, None) if not draw else (apis, tmp_img)
    
