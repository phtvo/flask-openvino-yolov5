from ai import ai 
from ai import utils

"""
Run a rest API exposing the yolov5s object detection model
"""
import argparse
import io
import os 

from PIL import Image
from flask import Flask, request, jsonify
import numpy as np 

import config 


app = Flask(__name__)
MODE = str(os.environ.get('FLASK_ENV', 'development')).strip()
if MODE == 'production':
    app.config.from_object(config.ProductionConfig)
    #print(config.ProductionConfig)
elif MODE == 'development':
    #print(config.DevelopmentConfig)
    app.config.from_object(config.DevelopmentConfig)



@app.route(config.DETECTION_URL_V1, methods=["POST"])
@app.route(config.DETECTION_URL_DEEPLY, methods=["POST"])
def predict(mode='d'):
    if not request.method == "POST":
        return

    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()

        img = np.asarray(Image.open(io.BytesIO(image_bytes)).convert('RGB') ) 
        results, image = model.process_form(img, deeply= True if mode =='d' else False)  # reduce size=320 for faster inference
        if image:
            image.save('tmp/tmp.png')
            image = utils.im_2_b64(image)
            
        return {"result" :{
            'items': results,
            'image': (image).decode('utf-8')
        }}


if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Flask API exposing AI model")
    #parser.add_argument("--port", default=5000, type=int, help="port number")
    #args = parser.parse_args()

    #model = ai(config.MODEL, img_size=config.IMAGE_SIZE)
    model = ai(
            config.MODEL, img_size=config.IMAGE_SIZE, 
            nclasses=config.NCLASSES, classes= config.CLASSES,
            conf= config.CONFIDENCE_THRESHOLD, iou= config.IOU_THRESHOLD,
            calibx= config.CALIBX, caliby= config.CALIBY
            )
    from waitress import serve
    serve(app, host=app.config['HOST'], port=app.config['PORT'])
