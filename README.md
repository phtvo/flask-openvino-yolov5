# Demo webform detection

```
usage: production: production.sh
```

* get current result [POST]
  * **/v1/ai/d** : detect chinh xac -> thoi gian lau hon
  * ~~**/v1/ai** : it chinh xac -> nhanh~~
  
  * Input parms : {"image" : file_image }
  * Output: trong Items chua text va vi tri cua box. e.i, "row": [[text, [xmin, ymin, xmax, ymax]], ...]
  ```json
    "result": {
        "items": {
            "row": list,
            "submit_button": list,
            "webform": list,
        },
        "image" : "string_base64"
      }
  ```

* Docker:
  * build: docker build -t webform . 
  * run: docker run -dp 5000:80  webform

* Example run: python [exmaple.py](example.py)
  * cai matplotlib (bash): pip install matplotlib
  * thay doi port trong DETECTION_URL
  * tay doi duong dan image trong TEST_IMAGE 