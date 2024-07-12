import cv2
import os
import numpy as np
from ultralytics import YOLO
import shutil
import argparse


class TrainModel:
    def __init__(self, model, epochs=10) -> None:
        self.model = YOLO(model)
        self.create_repo()
        self.create_yaml()
        self.deep_learning(epochs=epochs)
        
        
    def create_repo(self):
        main_folder = 'weights'


        # Créer le dossier principal s'il n'existe pas
        if not os.path.exists(main_folder):
            os.makedirs(main_folder)

            
        
    def create_yaml(self):
        train_path = os.path.join(os.getcwd(), 'datas/train')
        test_path = os.path.join(os.getcwd(), 'datas/test')
        valid_path = os.path.join(os.getcwd(), 'datas/val')
        
        data_yaml_content = f"""
        train: {train_path}
        test: {test_path}
        val: {valid_path}

        nc: 1
        
        names:
            0: elephant
        """
        
        with open('data.yaml', 'w') as f:
            f.write(data_yaml_content)
        print('data.yaml created')
        
    def deep_learning(self, epochs=5, datas='data.yaml'):
        results = self.model.train(data=datas, epochs=epochs, imgsz=320, lr0=0.01, lrf=0.1)
        result_dir = results.save_dir
        path = os.path.join(os.getcwd(), result_dir) + '/weights/best.pt'
        title = os.getcwd() + f'/weights/yolo_lr_{epochs}.pt'
        shutil.move(path, title)
        path = os.path.join(os.getcwd(), result_dir) + '/confusion_matrix.png'
        title = os.getcwd() + f'confusion_matrix_yolo_lr_{epochs}.png'
    
if __name__=='__main__':
    # parser = argparse.ArgumentParser(description='Creation model elephant')
    # parser.add_argument('--model', type=str, default=os.path.join(os.getcwd(), 'yolov8s.pt'),
    #                     help='Path to the model file')
    # parser.add_argument('--epochs', type=int, default=10,
    #                     help='Number of epochs to train the model')

    # args = parser.parse_args()

    #TrainModel(model=args.model, epochs=args.epochs)
    TrainModel(model='yolov8s.pt')