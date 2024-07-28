import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from imgbeddings import imgbeddings
from PIL import Image
import supervision as sv
import tensorflow as tf
from tensorflow.keras.losses import cosine_similarity
from applySam import SAM

class Pictures:
    def __init__(self, directory) -> None:
        self.sam = SAM()
        self.yolo_model = YOLO('yolov8s-seg.pt')
        self.dict = self.yolo_model.names
        paths = os.listdir(directory)
        self.arr_img, self.arr_title = [], []
        for path in paths:
            title = path.split('.')[0]
            title = './files/'+title
            path = os.path.join(directory, path)
            if 'groupe' in path:
                img = self.get_picture(path)
                if img is not None:
                    self.arr_img.append(img)
                    self.arr_title.append(title)
        self.treat_images()
        
    def get_picture(self, path):
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        image = None
        if ret:
            image = frame
        return image
            
    def treat_images(self):
        for title, img in zip(self.arr_title, self.arr_img):
            t = title + '_img.png'
            cv2.imwrite(t, img)
            mat_pts = self.sam.apply_sam(img)
            t = title + '_sam.npy'
            np.save(t, mat_pts)
            arr_cont = self.get_pred(img)
            if len(arr_cont) > 0:
                mat_cont = np.empty((len(arr_cont), 100, 2), int)
                for i, cont in enumerate(arr_cont):
                    cv2.polylines(img, [cont], True, (0, 0, 255), 2)
                    mat_cont[i] = cont
                   
            t = title + '_cont.npy'
            np.save(t, mat_cont)
                    
            
            h, w = mat_pts.shape[:2]
            for i in range(h):
                line = mat_pts[i]
                cv2.polylines(img, [line], True, (0, 0, 0), 2)
                
            cv2.imshow('', img)
            cv2.waitKey(1)
        
    def get_pred(self, frame):
        animals = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'person', 'surfboard', 'boat']
        animal_numbers = [key for key, value in self.dict.items() if value in animals]
        
        results = self.yolo_model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        arr_cont = []
        for detect in detections:
            mask = detect[1]
            name = detect[3]
            if name in animal_numbers:
                mask = np.array(mask*255).astype(np.uint8)
                mask = np.squeeze(mask)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contour = max(contours, key=cv2.contourArea)
                contour = self.sam.treat_contour(contour)
                arr_cont.append(contour)
        return arr_cont
        

    
        
class Picture:
    def __init__(self, path) -> None:
        self.path = path
        self.yolo_model = YOLO('yolov8s-seg.pt')
        self.ibed = imgbeddings()
        self.mask_annotator = sv.MaskAnnotator(opacity=1.0)
        self.get_picture()
        self.treat_picture()
        self.apply_sam()
        
         
        
    def get_picture(self):
        cap = cv2.VideoCapture(self.path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 150)
        ret, frame = cap.read()
        self.image = None
        if ret:
            self.image = frame
        cap.release()
        #cv2.imwrite('troupe_elephant.jpg', self.image)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        print('image', self.image.shape)
        plt.imshow(self.image)
        plt.show()
        
        
    def treat_picture(self):
        results = self.yolo_model(self.image)[0]
        detections = sv.Detections.from_ultralytics(results)
        print('detections', len(detections))
        detect = detections[0]
        mask = np.array(detect.mask*255).astype(np.uint8)
        mask = np.squeeze(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea)
        
        x, y, w, h = cv2.boundingRect(contour)
    
        cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        el_img = self.image[y:y+h, x:x+w]
        self.emb = self.get_embeddings(el_img)[0]
        print('emb', len(self.emb))
        
        cv2.polylines(self.image, [contour], True, (0, 0, 255), 2)
        cv2.imshow('', self.image)
        cv2.waitKey(0)
        
    
        
    def apply_sam(self):
        contours = np.load('./contours_sam.npy')
        for cont in contours:
            image = self.image.copy()
            line = np.array(cont)
            
            x, y, w, h = cv2.boundingRect(line)
            #cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            sub_img = image[y:y+h, x:x+w]
            emb = self.get_embeddings(sub_img)[0]
            sim = cosine_similarity(self.emb, emb)
            sim = float(sim)
            sim = abs(sim)
            sim = round(sim, 2)
            color = (255, 255, 255)
            if sim > 0.91:
                print('sim', sim)
                color = (0, 0, 255)
                cv2.polylines(self.image, [line], True, color, 2)
            
            
        cv2.imshow('', image)
        cv2.waitKey(0)
            
        
    def get_embeddings(self, image):
        pil_img = Image.fromarray(image)
        pil_img = pil_img.convert("RGB")
        return self.ibed.to_embeddings(pil_img)
        
        
if __name__=='__main__':
    directory = './videos'
    Pictures(directory)