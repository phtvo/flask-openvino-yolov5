FROM python:3.7.5-slim
RUN apt update && apt install -y libsm6 libxext6 && apt-get -y install g++
RUN apt-get -y install tesseract-ocr 
COPY . /app
WORKDIR /app
ENV TESSDATA_PREFIX=/app/tessdata
#RUN pip3 install cmake
#RUN pip3 install scikit-build
RUN pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir
RUN pip3 install easyocr==1.3.1
RUN pip3 install -r requirements.txt --no-cache-dir
CMD  [ "sh", "product.sh"]
