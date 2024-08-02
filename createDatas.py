import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Model.modelIA import Model

class Read:
    def __init__(self, video_path, model) -> None:
        self.path = video_path
        self.model = model
        self.read_video()
        
    def read_video(self):
        cap = cv2.VideoCapture(self.path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, desc="Traitement des frames")
        n = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                res = self.model.get_predictions(frame)
                if len(res) > 0:
                    self.write_data(res, n)
                pbar.update(1)
                
                n += 1
                
        cap.release()
        pbar.close()
        
    
    def write_data(self, res, n):
        title = f'Datas/frame_{n}.txt'
        with open(title, 'w') as f:
            for box, lab, score in res:
                x, y, w, h = box
                score = f'{score:.2f}'
                line = f'{lab} : {x} {y} {w} {h} {score}\n'
                f.write(line)
            
        
def main():
    video_path = 'Videos/safari.mp4'
    model = './Weights/yolov8s.pt'
    mod = Model(model)
    Read(video_path, mod)
        
        
if __name__=='__main__':
    main()
    