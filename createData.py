import cv2
import os
import numpy as np
from ultralytics import YOLO
import supervision as sv
import shutil

class TreatVideo:
    def __init__(self, input_path, model, data=False, seg=False) -> None:
        self.path = input_path
        self.model = YOLO(model)
        self.segment = seg
        self.bounding_box_annotator = sv.BoundingBoxAnnotator()
        self.mask_annotator = sv.MaskAnnotator(opacity=0.7)
        self.dict = self.model.names
        
        self.create_repo()
        self.read_video()
        if data:
            self.write_data()
        
    def create_repo(self):
        main_folder = 'datas'
        subfolders = ['train', 'val', 'test']

        # Créer le dossier principal s'il n'existe pas
        if not os.path.exists(main_folder):
            os.makedirs(main_folder)

        # Créer les sous-dossiers et les vider s'ils existent
        for subfolder in subfolders:
            path = os.path.join(main_folder, subfolder)
            if os.path.exists(path):
                self.clean_repo(path)
            else:
                os.makedirs(path)
                
    def clean_repo(self, folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Supprimer le fichier ou le lien symbolique
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Supprimer le dossier et son contenu
            except Exception as e:
                print(f'Erreur lors de la suppression {file_path}. Raison: {e}')

        
    def read_video(self):
        cap = cv2.VideoCapture(self.path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.dim = [width, height]
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n = 0
        self.arr_img, self.arr_contours, self.arr_rois = [], [], []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                image = frame.copy()
                
                frame, boxes, contours = self.get_predictions(frame)
                if not self.segment and len(boxes) > 0:
                    self.arr_img.append(image)
                    self.arr_rois.append(boxes)
                    
                if self.segment and len(contours) > 0:
                    self.arr_img.append(image)
                    self.arr_contours.append(contours)
                    
                    
                for box in boxes:
                    x, y, w, h = box
                    cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
                cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)
                
                cv2.imshow('', frame)
                
                key = cv2.waitKey(1)
                if key == 27:
                    break
                if n == self.total_frames - 1:
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
        animals = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']

        animal_numbers = [key for key, value in self.dict.items() if value in animals]
        
        arr_box, arr_cont = [], []
        for detect in detections:
            num = int(detect[3])
            if num in animal_numbers:
                box = np.array(detect[0]).astype(int)
                arr_box.append(box)
                mask = detect[1]
                if mask is not None:
                    mask = np.array(mask*255).astype(np.uint8)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)
                    for cont in contours:
                        arr_cont.append(cont)
        return annotated_frame, arr_box, arr_cont
    
        
    def write_contours(self, contours, title):
        w, h = self.dim
        with open(title, 'w') as f:
            for contour in contours:
                line = '0'
                for [p] in  contour:
                    x, y = p[0]/w, p[1]/h
                    x, y = round(x, 2), round(y, 2)
                    line += f' {x} {y}'
                line += '\n'
                f.write(line)
            
    def write_roi(self, rois, title):
        w, h = self.dim
        with open(title, 'w') as f:
            for roi in rois:
                x0, y0, x1, y1 = roi
                x, y = np.mean([x0, x1]), np.mean([y0, y1])
                x /= w
                y /= h
                width = (x1-x0)/w
                height = (y1-y0)/h
                x, y, width, height = round(x, 2), round(y, 2), round(width, 2), round(height, 2)
                line = f'0 {x} {y} {width} {height}\n'
                f.write(line)
        
    def write_data(self):
        directory = os.path.join(os.getcwd(), 'datas')
        d = int(0.7*len(self.arr_img))
        f = int(0.95*len(self.arr_img))
        
        print('nb_img', len(self.arr_img), d, f)
        path = os.path.join(directory, 'train')
        images = self.arr_img[:d]
        labels = self.arr_contours[:d]
        if not self.segment:
            labels = self.arr_rois[:d]
        
        for ind, (img, lab) in enumerate(zip(images, labels)):
            title = f'{path}/img_{ind}.png'
            cv2.imwrite(title, img)
            title = f'{path}/img_{ind}.txt'
            if self.segment:
                self.write_contours(lab, title)
            else:
                self.write_roi(lab, title)
                
        path = os.path.join(directory, 'test')
        images = self.arr_img[d:f]
        labels = self.arr_contours[d:f]
        if not self.segment:
            labels = self.arr_rois[d:f]
        for ind, (img, lab) in enumerate(zip(images, labels)):
            title = f'{path}/img_{ind}.png'
            cv2.imwrite(title, img)
            title = f'{path}/img_{ind}.txt'
            if self.segment:
                self.write_contours(lab, title)
            else:
                self.write_roi(lab, title)
                
        path = os.path.join(directory, 'val')
        images = self.arr_img[f:]
        labels = self.arr_contours[f:]
        if not self.segment:
            labels = self.arr_rois[d:f]
        for ind, (img, lab) in enumerate(zip(images, labels)):
            title = f'{path}/img_{ind}.png'
            cv2.imwrite(title, img)
            title = f'{path}/img_{ind}.txt'
            if self.segment:
                self.write_contours(lab, title)
            else:
                self.write_roi(lab, title)
                
                
            
        


    
        
        
if __name__=='__main__':
    paths = os.listdir(os.path.join(os.getcwd(), 'videos'))
    path = os.path.join(os.getcwd(), 'videos', 'elephant.mp4')
    path = os.path.join(os.getcwd(), 'videos', path)
    model = 'yolov8s.pt'
    TreatVideo(input_path=path, model=model, data=True, seg=False)
    