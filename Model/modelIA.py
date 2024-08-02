import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

class Model:
    def __init__(self, model) -> None:
        self.model = YOLO(model)
        
    def get_predictions(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        animals = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
        result = []
        for detect in detections:
            score = detect[2]
            num = int(detect[3])
            label = self.model.names[num]
            if label in animals:
                box = np.array(detect[0]).astype(int)
                result.append([box, label, score])
        return result
    
    def draw_predictions(self, frame):
        res = self.get_predictions(frame)
        for box, lab in res:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            p = (x, y-10)
            self.put_text(frame, lab, p)
            
    def put_text(self, frame, texte, p):
        font = 1
        font_scale = 1.0
        color = (0, 255, 0)
        thick = 1
        cv2.putText(frame, texte, p, font, font_scale, color, thick)

            
            