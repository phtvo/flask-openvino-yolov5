"""Perform test request"""
import pprint

import requests, io
import base64

import matplotlib.pyplot as plt 

DETECTION_URL = "http://localhost:5000/v1/ai/d"
TEST_IMAGE = r"F:\Work\contactform\data\test\19333_.png"

from PIL import Image

# Take in base64 string and return cv image
def stringToRGB(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    image = Image.open(io.BytesIO(imgdata))
    return image

image_data = open(TEST_IMAGE, "rb").read()

response = requests.post(DETECTION_URL, files={"image": image_data}).json()

image = stringToRGB(response['result']['image'])

plt.imshow(image)
plt.show()