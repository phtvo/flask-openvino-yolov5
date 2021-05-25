import easyocr

class OCR:
    def __init__(self, gpu=False) -> None:
        self.reader = easyocr.Reader(['ja', 'en'], gpu= gpu)

    def topleft_text(self, image):
        """Return topleft text of image

        Args:
            image (np.ndarray): image

        Returns:
            str: text
        """
        result = self.reader.readtext(image)
        if result:
            result.sort(key=lambda x : (sum(x[0][0])) )
            text = result[0][-2]
            if len(result) == 2:
                text0_y = result[0][0][0][1]
                text1_y = result[1][0][0][1] 
                if abs(text0_y - text1_y) <= 10:
                    text = result[0][-2] + '_' + result[0][-2]

            return text
        else:
            return ''

