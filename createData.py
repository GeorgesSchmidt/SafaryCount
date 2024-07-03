import cv2
import os
import numpy as np
from ultralytics import YOLO
import supervision as sv
import shutil
import argparse

class TreatVideo:
    def __init__(self, input_path, model) -> None:
        self.path = input_path
        self.model = YOLO(model)
        self.bounding_box_annotator = sv.BoundingBoxAnnotator()
        self.mask_annotator = sv.MaskAnnotator(opacity=0.7)
        self.dict = self.model.names
        
        self.create_repo()
        self.read_video()
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

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = self.write_datas(frame, n)
                cv2.imshow('', frame)
                
                key = cv2.waitKey(1)
                if key == 27:
                    break
                if n == self.total_frames - 5:
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
        animals = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']

        animal_numbers = [key for key, value in self.dict.items() if value in animals]
        
        arr_box, arr_cont = [], []
        for detect in detections:
            num = int(detect[3])
            if num in animal_numbers:
                box = np.array(detect[0]).astype(int)
                arr_box.append(box)
        return annotated_frame, arr_box
    
    def write_datas(self, frame, num):
        directory = os.path.join(os.getcwd(), 'datas')
        image, arr_box = self.get_predictions(frame)
        print('box', len(arr_box))
        if len(arr_box) > 0:
            sub_direct = 'train'
            if num > 0.7*self.total_frames and num < 0.85*self.total_frames:
                sub_direct = 'test'
            if num > 0.85*self.total_frames:
                sub_direct = 'val'
                
            path = os.path.join(directory, sub_direct)
            title = os.path.join(path, f'img_{num}.png')
            cv2.imwrite(title, frame)
            
            title = os.path.join(path, f'img_{num}.txt')
            self.write_roi(arr_box, title)
        return image
        
            
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
        


        
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Traitement vidéo avec YOLO.')
    parser.add_argument('input_path', type=str, help='Chemin de la vidéo à traiter')
    parser.add_argument('model', type=str, help='Chemin du modèle YOLO')

    args = parser.parse_args()

    TreatVideo(input_path=args.input_path, model=args.model)
    