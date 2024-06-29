import os
import ultralytics
import numpy as np
from io import BytesIO
from PIL import Image


class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


    
    def predict(self):
        # load model
        model = ultralytics.YOLO(os.path.join("model","best.pt"))

        imagename = self.filename
        image = Image.open(imagename)
        image_array = np.array(image)
        result = model(image_array)
        print(result[0].probs.data)

        if result[0].probs.data[1] >= 0.5:
            prediction = 'smoking'
            return [{ "image" : prediction}]
        else:
            prediction = 'not_smoking'
            return [{ "image" : prediction}]