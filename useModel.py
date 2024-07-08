import cv2
import os
import numpy as np
from ultralytics import YOLO
import supervision as sv
from motpy import Detection, MultiObjectTracker

class UseModel:
    def __init__(self, model, video_path) -> None:
        self.path = video_path
        self.model = YOLO(model)
        self.bounding_box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.tracker = sv.ByteTrack()
        self.read_video()
        self.get_results()
        
    def read_video(self):
        cap = cv2.VideoCapture(self.path)
        cap_fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.dim = [width, height]
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.total_frames -= 1
        self.detects = [0]*self.total_frames
        
        tracker = MultiObjectTracker(dt=1 / cap_fps, tracker_kwargs={'max_staleness': 10})
        self.result_track = {}
        id_dict = {}
        j = 0
        n = 0
        for i in range(self.total_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                #frame = self.get_annoneted_img(frame, i)
                rois = self.get_pred(frame)
                detect = []
                for roi in rois:
                    detect.append(Detection(box=roi, score=1.0, class_id=0))
                    
                self.put_texte(frame, result=len(detect))
                    
                tracker.step(detections=detect)
                track_results = tracker.active_tracks()
                id_dict, j = self.update_id_dict(id_dict, j, track_results)
                self.show_result(frame, track_results, id_dict, num_img=i)
                
                
                cv2.imshow('', frame)
                
                key = cv2.waitKey(1)
                if key == 27:
                    break
                
        cap.release()
        cv2.destroyAllWindows()
        
    def show_result(self, frame, track_results, id_dict, num_img):
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (0, 128, 128), (128, 0, 128), (192, 192, 192), (128, 128, 128), (64, 64, 64), (255, 165, 0), (255, 192, 203), (75, 0, 130), (123, 104, 238), (255, 20, 147)]
        font = 1
        font_scale = 1.0
        for ind, object in enumerate(track_results):
            x, y, w, h = object.box
            x, y, w, h = int(x), int(y), int(w), int(h)
            object_id = object.id
            confidence = object.score
            confidence = round(confidence, 2)
            num = id_dict[object_id]
            
            
            color = colors[ind]
            cv2.rectangle(frame, (x, y), (w, h), color, 2)
            texte = f'ind {num}_{confidence}'
            if num not in self.result_track:
                self.result_track[num] = []
            else:
                self.result_track[num].append(num_img)
                
            p = [x, y-10]
            cv2.putText(frame, texte, p, font, font_scale, color, 1)
            
            
    def get_annoneted_img(self, frame, num):
        results = self.model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        nb_detect = len(detections)
        self.detects[num] = nb_detect
        self.put_texte(frame, nb_detect)
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
    
    def get_pred(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        boxes = []
        for detect in detections:
            num = int(detect[3])
            box = np.array(detect[0]).astype(int)
            boxes.append(box)
        return boxes
        

    def put_texte(self, frame, result=0):
        texte = f'{result} detections'
        p = (10, 50)
        font = 1
        font_scale = 1.0
        color = (0, 0, 0)
        thick = 1
        cv2.putText(frame, texte, p, font, font_scale, color, thick)


    def update_id_dict(self, id_dict, j, track_results):
        # Update ID dictionary with new track results
        for track_result in track_results:
            if track_result.id not in id_dict:
                id_dict[track_result.id] = j
                j += 1
        return id_dict, j

        
    def get_results(self):
        ind = self.result_track.keys()
        print('total number of images :', self.total_frames)
        num = range(self.total_frames)
        num_no = []
        for n in num:
            if n not in ind:
                num_no.append(n)
                
        print(len(num_no), ' images with no elephants')
        
                
        print('number of elephants detected ', len(ind))
        for ind, val in self.result_track.items():
            print(f'elephant n° {ind} : image {val[0]} - image {val[-1]}')
    

if __name__=='__main__':
    
    path = os.path.join(os.getcwd(), 'videos', 'elephant.mp4')

    model = os.path.join(os.getcwd(), 'elephant_epochs_retrained_10_10.pt')
    UseModel(model=model, video_path=path)