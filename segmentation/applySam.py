import sys
import os
sys.path.append(os.path.abspath('segment-anything'))

from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import cv2
import numpy as np
from math import tau


class SAM:
    def __init__(self) -> None:
        sam = sam_model_registry["vit_h"](checkpoint="/Users/georgesschmidt/VisualCodeProjects/SafaryCount/segmentation/sam_vit_h_4b8939.pth")
        self.mask_generator = SamAutomaticMaskGenerator(sam)
        
    def apply_sam(self, image):
        masks = self.mask_generator.generate(image)
        h, w = image.shape[:2]
        black = np.zeros((h, w), dtype=np.uint8)
        black.fill(255)
        h = len(masks)
        w = 100
        mat_pts = np.empty((h, w, 2), int)
        m = masks[0]
        print(m)
        for i in range(len(masks)):
            m = masks[i]
            mask = masks[i]['segmentation']
            mask = (mask.astype(np.uint8)) * 255
            contour = self.get_contours(mask)
            contour = self.treat_contour(contour)
            
            mat_pts[i] = contour
            cv2.polylines(black, [contour], True, (0), 2)
        
        
        return black
            
            
    
    def get_contours(self, img):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea)
        return contour
    
    def treat_contour(self, contour):
        x = [v[0] for [v] in contour]
        y = [v[1] for [v] in contour]
        angle = np.linspace(0, tau, len(x))
        xvals = np.linspace(0, tau, 100)
        vx = np.interp(xvals, angle, x).astype(int)
        vy = np.interp(xvals, angle, y).astype(int)
        return np.array(list(zip(vx, vy)))
        
        
            
            
def get_image_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    _, frame = cap.read()
    cap.release()
    title = '/Users/georgesschmidt/VisualCodeProjects/SafaryCount/pictures/elephant_lake.png'
    cv2.imwrite(title, frame)
    
        
        
        
if __name__=='__main__':
    path_img = '/Users/georgesschmidt/VisualCodeProjects/SafaryCount/pictures/elephant_lake.png'
    sam = SAM()
    img = cv2.imread(path_img)
    result = sam.apply_sam(img)
    cv2.imshow('', result)
    cv2.waitKey(0)
    
    title = '/Users/georgesschmidt/VisualCodeProjects/SafaryCount/pictures/elephant_lake_result_sam.png'
    
    cv2.imwrite(title, result)

