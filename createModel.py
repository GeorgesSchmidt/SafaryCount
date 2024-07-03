import cv2
import os
import numpy as np
from ultralytics import YOLO
import shutil
import argparse

class TrainModel:
    def __init__(self, model, epochs=10) -> None:
        self.model = YOLO(model)
        self.create_yaml()
        self.deep_learning(epochs=epochs)
        
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

        # Écrire le contenu dans le fichier data.yaml
        title = os.path.join(os.getcwd(), 'data.yaml')
        with open(title, 'w') as f:
            f.write(data_yaml_content)

        print('data.yaml created')
        
    def deep_learning(self, epochs=20, datas='data.yaml'):
        results = self.model.train(data=datas, epochs=epochs, imgsz=320)
        result_dir = results.save_dir
        path = os.path.join(os.getcwd(), result_dir) + '/weights/best.pt'
        title = os.getcwd() + f'/elephant_model_epochs_{epochs}.pt'
        shutil.move(path, title)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Creation model elephant')
    parser.add_argument('--model', type=str, default=os.path.join(os.getcwd(), 'yolov8s.pt'),
                        help='Path to the model file')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train the model')

    args = parser.parse_args()

    TrainModel(model=args.model, epochs=args.epochs)