
############### Object detection (yolo-openvino) Config #######################
CONFIDENCE_THRESHOLD = 0.55
IOU_THRESHOLD = 0.17
MODEL= 'data/openvino/contactform/contactform.xml'
IMAGE_SIZE = 640
CALIBX = 30
CALIBY = 10
CLASSES = [
'webform',
'header',
'footer',
'row',
'submit_button',
'terms',
]
NCLASSES = len(CLASSES)
############### Config esasy ocr #######################
EASY_OCR = True
OCR_GPU = False
############### API routes #######################
DETECTION_URL_DEEPLY = "/v1/ai/<string:mode>"
DETECTION_URL_V1 = "/v1/ai"

############### Flask Config ########################

class Config(object):
    DEBUG = False
    TESTING = False
    CSRF_ENABLED = True
    JSON_SORT_KEYS = False
    #SECRET_KEY = 'this-really-needs-to-be-changed'


class ProductionConfig(Config):
    DEBUG = False
    PORT = 80
    HOST = '0.0.0.0'

class StagingConfig(Config):
    DEVELOPMENT = True
    DEBUG = True


class DevelopmentConfig(Config):
    DEVELOPMENT = True
    DEBUG = True
    PORT= 5000
    HOST = 'localhost'


class TestingConfig(Config):
    TESTING = True