import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import supervision as sv

class DataSeg:
    def __init__(self, path) -> None:
        self.path = path
        self.model = YOLO('yolov8s-seg.pt')
        self.mask_annotator = sv.MaskAnnotator(opacity=1.0)
        self.read_movie()
        
    def read_movie(self):
        cap = cv2.VideoCapture(self.path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for i in range(total_frames):
            ret, frame = cap.read()
            if ret:
                self.treat_picture(frame)
                cv2.imshow('', frame)
                key = cv2.waitKey(1)
                if key == 27:
                    break
        cap.release()
        cv2.destroyAllWindows()
        
    def treat_picture(self, frame):
        results = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        for detect in detections:
            mask = np.array(detect[1]*255).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea)
            cv2.polylines(frame, [contour], True, (0, 0, 255), 2)
        
    
        
if __name__=='__main__':
    path_video = os.path.join(os.getcwd(), 'videos', 'elephant_1.mp4')
    DataSeg(path_video)
        