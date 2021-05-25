from time import time
from ai import ai 
import cv2 
import matplotlib.pyplot as plt 

import config 

model = ai(
            config.MODEL, img_size=config.IMAGE_SIZE, 
            nclasses=config.NCLASSES, classes= config.CLASSES,
            conf= config.CONFIDENCE_THRESHOLD, iou= config.IOU_THRESHOLD
            )

for _ in range(5):
    start = time()
    image = cv2.imread('test_image.png')
    res , im = model.process_form(image, draw=True, deeply= True)
    end = time()
    print(f"Process takes {round(1000*(end - start), 2)}ms")
    print('-----------')


plt.imshow(im)
plt.show()