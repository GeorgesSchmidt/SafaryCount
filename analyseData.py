import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import re

class Analyse:
    def __init__(self) -> None:
        self.directory = 'Datas'
        self.path = 'Videos/safari.mp4'
        self.read_datas()
        self.detections_animals()
        
    def read_datas(self):
        cap = cv2.VideoCapture(self.path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        paths = os.listdir(self.directory)
        values = []
        for i in range(total_frames):
            title = f'frame_{i}.txt'
            val = 0
            if title in paths:
                cont = self.read_file(title)
                val = len(cont)
                
            values.append(val)
            
        intervalles = self.find_intervalles(values)
        
        print('sequences with no detections :')
        print(intervalles)
        title = 'Datas/intervals_no_detect.txt'
        with open(title, 'w') as f:
            for inter in intervalles:
                line = f'{inter[0]} {inter[1]}\n'
                f.write(line)
                
        plt.figure(figsize=(15, 5))
        plt.title(f'detections in {self.path}')
        plt.plot(values)
        plt.savefig('Pictures/detections.png')
                
    def find_intervalles(self, values):
        valeurs = np.array(values)
        is_zero = (valeurs == 0)
        zero_diff = np.diff(is_zero.astype(int))
        start_indices = np.where(zero_diff == 1)[0] + 1
        end_indices = np.where(zero_diff == -1)[0]
        
        # Handle the case where the sequence starts or ends with zeros
        if is_zero[0]:
            start_indices = np.insert(start_indices, 0, 0)
        if is_zero[-1]:
            end_indices = np.append(end_indices, len(valeurs) - 1)
        
        intervals = list(zip(start_indices, end_indices))
        # Filtrer les intervalles nuls
        intervals = [(start, end) for start, end in intervals if start != end]
        
        return intervals
    
    def detections_animals(self):
        cap = cv2.VideoCapture(self.path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        animals = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
        paths = os.listdir(self.directory)
        
        animal_images = {animal: [] for animal in animals}

        # Collecte des données pour chaque animal
        for path in paths:
            content = self.read_file(path)
            for line in content:
                lab = line.split(' ')[0]
                if lab in animal_images:
                    img = path.split('.')[0]
                    num = int(img.split('_')[-1])
                    animal_images[lab].append(num)

        # Filtrage des animaux détectés
        detected_animals = {animal: images for animal, images in animal_images.items() if images}
        fig, axs = plt.subplots(len(detected_animals), 1, figsize=(20, 10), sharex=True)
        axs = axs.flatten()  # Convertir en tableau plat pour un accès plus facile
        for ax in axs:
            ax.set_xlim(0, total_frames)
            ax.set_ylim(0, 20)
        
        # Tracer les graphiques pour les animaux détectés
        for ind, (animal, images) in enumerate(detected_animals.items()):
            nb, bins = np.histogram(images, bins=range(total_frames + 1))
            axs[ind].bar(bins[:-1], nb, width=1)
            axs[ind].set_ylabel(animal)  # Utiliser le nom de l'animal comme label pour l'axe Y
        
        plt.xlabel('Frame Number')
        plt.tight_layout()  # Ajuster la mise en page pour éviter les chevauchements
        #plt.show()
        plt.savefig('Pictures/detect_animals.png')
        
            
    def read_file(self, path):
        path = os.path.join('Datas', path)
        with open(path, 'r') as f:
            content = f.readlines()
        
        return content
            
if __name__=='__main__':
    Analyse()
            
        