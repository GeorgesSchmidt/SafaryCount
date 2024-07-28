import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

import supervision as sv

class Find:
    def __init__(self, path) -> None:
        self.image = cv2.imread(path)
        self.model = YOLO('yolov8s.pt')
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        plt.imshow(self.image)
        plt.show()
        
        self.treat()
        
    def treat(self):
        results = self.model(self.image)[0]
        print('result', len(results))
        pred = results[0]
        print(pred)
        
        
if __name__=='__main__':
    path_img = '/Users/georgesschmidt/VisualCodeProjects/SafaryCount/segmentation/troupe_elephant_1.jpg'
    Find(path_img)