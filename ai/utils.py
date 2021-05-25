from PIL import Image
from io import BytesIO
import base64
# Convert Image to Base64 
def im_2_b64(image):
    buff = BytesIO()
    image = image.convert('RGB')
    image.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue())
    return img_str