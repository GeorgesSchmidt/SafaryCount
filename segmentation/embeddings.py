import cv2
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from imgbeddings import imgbeddings
from PIL import Image
import tensorflow as tf
from tensorflow.keras.losses import cosine_similarity

class Emb:
    def __init__(self) -> None:
        self.ibed = imgbeddings()
        paths = os.listdir('./files')
        self.datas = []
        for i in range(1, 14):
            sub = []
            for path in paths:
                num = path.split('_')[2]
                num = int(num)
                if num == i:
                    sub.append(path)
            self.datas.append(sub)
        self.calculate_emb()
        
    def calculate_emb(self):
        self.embeddings = []
        for paths in self.datas:
            paths_files = [p for p in paths if p.endswith('.npy')]
            path_cont = None
            for p in paths_files:
                if 'cont' in p:
                    path_cont = os.path.join('./files', p)
            pts_cont = np.load(path_cont)
            [path_img] = [p for p in paths if p.endswith('.png')]
            path_img = os.path.join('./files', path_img)
            img = cv2.imread(path_img)
            image = img.copy()
            boxes = self.get_roi(pts_cont)
            for box in boxes:
                x, y, w, h = box
                face = image[y:y+h, x:x+w]
                emb = self.get_embeddings(face)
                self.embeddings.append(emb)
            self.draw_mat(img, pts_cont, color=(0, 0, 255))
            cv2.imshow('', img)
            cv2.waitKey(0)
            
    def create_svc(self):
        pass
        
    def treat(self):
        for paths in self.datas:
            paths_files = [p for p in paths if p.endswith('.npy')]
            path_sam = None
            path_cont = None
            for p in paths_files:
                if 'sam' in p:
                    path_sam = os.path.join('./files', p)
                if 'cont' in p:
                    path_cont = os.path.join('./files', p)
           
            pts_sam = np.load(path_sam)
            pts_cont = np.load(path_cont)
            
            [path_img] = [p for p in paths if p.endswith('.png')]
            path_img = os.path.join('./files', path_img)
            img = cv2.imread(path_img)
            image = img.copy()
            self.draw_mat(img, pts_sam)
            self.draw_mat(img, pts_cont, color=(0, 0, 255))
            
            boxes = self.get_roi(pts_cont)
            embeddings = []
            for box in boxes:
                x, y, w, h = box
                face = image[y:y+h, x:x+w]
                emb = self.get_embeddings(face)
                embeddings.append(emb)
                
            boxes = self.get_roi(pts_sam)
            for box in boxes:
                x, y, w, h = box
                face = image[y:y+h, x:x+w]
                emb = self.get_embeddings(face)
                for e in embeddings:
                    sim = cosine_similarity(e, emb)
                    sim = float(sim)
                    sim = abs(sim)
                    sim = round(sim, 2)
                    if sim > 0.93:
                        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
                    
            cv2.imshow('', img)
            cv2.waitKey(0)
            
    def draw_mat(self, frame, mat, color=(0, 0, 0)):
        h = mat.shape[0]
        for i in range(h):
            line = mat[i]
            cv2.polylines(frame, [line], True, color, 2)
                
        
    def get_roi(self, mat):
        h = mat.shape[0]
        boxes = []
        for i in range(h):
            line = mat[i]
            boxes.append(cv2.boundingRect(line))
        return boxes
   
    def get_embeddings(self, image):
        pil_img = Image.fromarray(image)
        pil_img = pil_img.convert("RGB")
        return self.ibed.to_embeddings(pil_img)


        
if __name__=='__main__':
    Emb()