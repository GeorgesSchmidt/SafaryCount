import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from imgbeddings import imgbeddings
from PIL import Image
import tensorflow as tf
from tensorflow.keras.losses import cosine_similarity

class detect_elephant:
    def __init__(self, path_img, path_img_sam, model) -> None:
        self.ibed = imgbeddings()
        self.img = cv2.imread(path_img)
        self.img_sam = cv2.imread(path_img_sam, cv2.IMREAD_GRAYSCALE)
        self.model = YOLO(model)
        self.get_prediction()
        self.get_boxes_sam()
        
        
    def get_prediction(self):
        image = self.img.copy()
        results = self.model(self.img, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        boxes = detections.xyxy
        self.embed_yolo = []
        for box in boxes:
            x, y, w, h = np.array(box).astype(int)
            cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)
            img = self.img[y:h, x:w]
            emb = self.get_embeddings(img)[0]
            self.embed_yolo.append(emb)
        #cv2.imshow('', image)
        #cv2.waitKey(0)
        print('emb yolo', len(self.embed_yolo))
        
    def get_boxes_sam(self):
        _, bin = cv2.threshold(self.img_sam, 0, 255, cv2.THRESH_BINARY)
        bin = cv2.bitwise_not(bin)
        contours, _ = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        color = self.img.copy()
        self.emb_sam = []
        for contour in contours:
            box = cv2.boundingRect(contour)
            x, y, w, h = box
            img = self.img[y:y+h, x:x+w]
            emb = self.get_embeddings(img)[0]
            self.emb_sam.append(emb)
            
            cv2.rectangle(color, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #cv2.imshow('', color)
        #cv2.waitKey(0)
        print('emb sam', len(self.emb_sam))
        
    def get_embeddings(self, image):
        pil_img = Image.fromarray(image)
        pil_img = pil_img.convert("RGB")
        return self.ibed.to_embeddings(pil_img)
        
        
if __name__=='__main__':
    path_img = '/Users/georgesschmidt/VisualCodeProjects/SafaryCount/pictures/elephant_lake.png'
    path_img_sam = '/Users/georgesschmidt/VisualCodeProjects/SafaryCount/pictures/elephant_lake_result_sam.png'
    model = '/Users/georgesschmidt/VisualCodeProjects/SafaryCount/weights/yolo_equitable_100.pt'
    detect_elephant(path_img, path_img_sam, model)
    
        