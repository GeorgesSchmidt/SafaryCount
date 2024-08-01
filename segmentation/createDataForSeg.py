import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

class Pictures:
    def __init__(self) -> None:
        self.directory = '../PicSeg'
        paths = os.listdir('./videos')
        self.arr_img = []
        for path in paths:
            if 'groupe' in path:
                p = os.path.join('./videos', path)
                img = self.get_first_img(p)
                self.arr_img.append(img)
                
        for ind, img in enumerate(self.arr_img):
            title = f'./PicSeg/img_{ind}.png'
            cv2.imwrite(title, img)
                
        
    def get_first_img(self, path):
        cap = cv2.VideoCapture(path)
        _, frame = cap.read()
        cap.release()
        
        return frame
    
if __name__=='__main__':
    Pictures()