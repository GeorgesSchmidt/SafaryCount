import cv2
import os
import numpy as np
from ultralytics import YOLO
import supervision as sv

class UseModel:
    def __init__(self, model, video_path) -> None:
        self.path = video_path
        self.model = YOLO(model)
        self.bounding_box_annotator = sv.BoundingBoxAnnotator()
        self.read_video()
        
    def read_video(self):
        cap = cv2.VideoCapture(self.path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.dim = [width, height]
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = self.get_annoneted_img(frame)
                cv2.imshow('', frame)
                
                key = cv2.waitKey(1)
                if key == 27:
                    break
                if n == self.total_frames - 1:
                    break
                n+=1
        cap.release()
        cv2.destroyAllWindows()
        
    def get_annoneted_img(self, frame):
        results = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(detections['class_name'], detections.confidence)
        ]
        
        
        annotated_frame =self.label_annotator.annotate(  # type: ignore
        scene=frame.copy(), detections=detections, labels=labels
        )
        
        annotated_frame =self.bounding_box_annotator.annotate(
        scene=annotated_frame.copy(),
        detections=detections
        ) 
        return annotated_frame



        
        

if __name__=='__main__':
    paths = os.listdir(os.path.join(os.getcwd(), 'videos'))
    path = os.path.join(os.getcwd(), 'videos', 'elephant.mp4')
    path = os.path.join(os.getcwd(), 'videos', path)
    model = 'elephant_model_epochs_3.pt'
    UseModel(model=model, video_path=path)