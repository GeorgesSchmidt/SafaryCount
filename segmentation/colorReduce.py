import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

class Contours:
    def __init__(self, path) -> None:
        self.path = path
        self.read_movie()
        
    def read_movie(self):
        cap = cv2.VideoCapture(self.path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for i in range(total_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                #frame = self.treat_image(frame, 2)
                frame = self.get_contours(frame)
                cv2.imshow('', frame)
                key = cv2.waitKey(1)
                if key == 27:
                    break
        cap.release()
        cv2.destroyAllWindows()
        
    def treat_image(self, image, bits_per_channel):
        factor = 256 // (2 ** bits_per_channel)
    
        # Appliquer la réduction de couleur
        reduced_image = (image // factor) * factor + factor // 2
        
        return reduced_image
    
    def get_contours(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image, 100, 350)
        return edges
        
        
      
if __name__=='__main__':
    path = './videos/elephant_1.mp4'
    Contours(path)