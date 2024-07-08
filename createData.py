import cv2
import os
import numpy as np
from ultralytics import YOLO
import supervision as sv
import shutil
import uuid
import argparse
from tqdm import tqdm


class TreatVideo:
    def __init__(self, path_no, path_yes, model) -> None:
        if model is not None:
            self.model = YOLO(model)
        else:
            self.model = YOLO('yolov8s.pt')
            
        self.bounding_box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.mask_annotator = sv.MaskAnnotator(opacity=0.7)
        self.dict = self.model.names
        
        self.create_repo()
        
        path_no = os.path.join(os.getcwd(), 'videos', 'no_elephant_1.mp4')
        self.elephant = False
        print('no elephant')
        self.treat_video(path_no)
        
        path_yes = os.path.join(os.getcwd(), 'videos', 'elephant_1.mp4')
        self.elephant = True
        print('with elephant')
        self.treat_video(path_yes)
        
        self.check_datas()
        
        
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

        
    def treat_video(self, path):
        cap = cv2.VideoCapture(path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.dim = [width, height]
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_frames -= 1
        pbar = tqdm(total=self.total_frames, desc="Traitement des frames")
        arr_count = 0
        for i in range(self.total_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                self.write_datas(frame, i)
                boxes = self.get_predictions(frame)
                if len(boxes) > 0:
                    arr_count += 1                
            pbar.update(1)
                
        cap.release()
        pbar.close()
    
        print('result ', arr_count, ' / ', self.total_frames)
       
            
    
    def get_predictions(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        
        animals = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']

        animal_numbers = [key for key, value in self.dict.items() if value in animals]
        
        res = []
        for detect in detections:
            num = int(detect[3])
            box = np.array(detect[0]).astype(int)
            if num in animal_numbers:
                res.append(box)
        return res
    
    def write_datas(self, frame, num):
        directory = os.path.join(os.getcwd(), 'datas')
        
        sub_direct = ''
        if num <= 0.7*self.total_frames:
            sub_direct = 'train'
        if num > 0.7*self.total_frames and num < 0.85*self.total_frames:
            sub_direct = 'test'
        if num > 0.85*self.total_frames:
            sub_direct = 'val'


        path = os.path.join(directory, sub_direct)
        t = uuid.uuid4().hex[:6]
       
        title = os.path.join(path, f'elephant_{t}.png')
        if not self.elephant:
            title = os.path.join(path, f'no_elephant_{t}.png')
        cv2.imwrite(title, frame)

        title = os.path.join(path, f'elephant_{t}.txt')
        if not self.elephant:
            title = os.path.join(path, f'no_elephant_{t}.txt')

        lines = []
        if self.elephant:
            pred = self.get_predictions(frame)
            height, width = frame.shape[:2]
            for box in pred:
                x0, y0, x1, y1 = box
                c_x = np.mean([x0, x1])/width
                c_y = np.mean([y0, y1])/height
                w = (x1-x0)/width
                h = (y1-y0)/height
                c_x, c_y = round(c_x, 2), round(c_y, 2)
                w, h = round(w, 2), round(h, 2)
                line = f'0 {c_x} {c_y} {w} {h}\n'
                lines.append(line)
            
        with open(title, 'w') as f:
            for line in lines:
                f.write(line)
                
    def check_datas(self):
        directory = os.path.join(os.getcwd(), 'datas')
        sub_dir = os.listdir(directory)
        n = 0
        nb = 0
        print(sub_dir)
        for dir in sub_dir:
            paths = os.listdir(os.path.join(directory, dir))
            for path in paths:
                if 'no_elephant' in path:
                    nb += 1
                else:
                    n += 1
        print("datas")
        print(f'{n} images with {n} labels with elephants')
        print(f'{nb} images with {nb} labels with no elephants')
                
        
                
            
 

        
if __name__=='__main__':
    # parser = argparse.ArgumentParser(description='Traitement vidéo avec YOLO.')
    # parser.add_argument('input_path', type=str, help='Chemin de la vidéo à traiter')
    # parser.add_argument('model', type=str, help='Chemin du modèle YOLO')

    # args = parser.parse_args()

    #TreatVideo(input_path=args.input_path, model=args.model)
    path_no = os.path.join(os.getcwd(), 'videos', 'no_elephant_1.mp4')
    path_yes = os.path.join(os.getcwd(), 'videos', 'elephant_1.mp4')

    TreatVideo(path_no, path_yes)
    