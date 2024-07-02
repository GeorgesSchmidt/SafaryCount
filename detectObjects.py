import cv2
import os
import numpy as np
from ultralytics import YOLO
import supervision as sv

class TreatVideo:
    def __init__(self, input_path) -> None:
        self.path = input_path
        self.model = YOLO('yolov8s-seg.pt')
        self.bounding_box_annotator = sv.BoundingBoxAnnotator()
        self.mask_annotator = sv.MaskAnnotator(opacity=0.7)
        print(self.model.names)
        self.read_video()
        
    def read_video(self):
        cap = cv2.VideoCapture(self.path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                
                frame, boxes, contours = self.get_predictions(frame)
                
                for box in boxes:
                    x, y, w, h = box
                    cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
                cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)
                
                cv2.imshow('', frame)
                
                key = cv2.waitKey(1)
                if key == 27:
                    break
                if n == total_frames - 1:
                    break
                n+=1
        cap.release()
        cv2.destroyAllWindows()
        
    def get_predictions(self, frame):
        results = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        annotated_frame =self.bounding_box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
        ) 
        annotated_frame = self.mask_annotator.annotate(
        scene=frame.copy(),
        detections=detections
        )
        arr_box, arr_cont = [], []
        for detect in detections:
            num = int(detect[3])
            if num == 20:
                box = np.array(detect[0]).astype(int)
                arr_box.append(box)
                mask = np.array(detect[1]*255).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)
                for cont in contours:
                    arr_cont.append(cont)
        return annotated_frame, arr_box, arr_cont
    
    

    
        
        
if __name__=='__main__':
    paths = os.listdir(os.path.join(os.getcwd(), 'videos'))
    path = os.path.join(os.getcwd(), 'videos', 'elephant.mp4')
    path = os.path.join(os.getcwd(), 'videos', path)
    TreatVideo(path)
    