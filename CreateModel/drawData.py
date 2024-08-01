import cv2
import numpy as np
import random
import os
import shutil

class VideoRectangleDrawer:
    def __init__(self, video_path, num_images=30):
        self.video_path = video_path
        self.num_images = num_images
        self.window_name = 'drawing rois'
        self.drawing = False
        self.start_x, self.start_y = -1, -1
        self.current_image_index = 0

        # Liste pour conserver les rectangles dessinés
        self.rectangles = []
        self.clean_repo()
        # Charger la vidéo
        self.capture = cv2.VideoCapture(video_path)
        if not self.capture.isOpened():
            raise ValueError("Impossible d'ouvrir la vidéo")

        # Obtenir le nombre total de frames
        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # Extraire les images aléatoires
        self.frame_indices = sorted(random.sample(range(self.total_frames), num_images))
        self.current_frame = None
        self.load_frame(self.frame_indices[self.current_image_index])

        # Créer une fenêtre et assigner la fonction de rappel pour les événements de souris
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.draw_rectangle, param=self)
        
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


    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x, self.start_y = x, y
            self.temp_image = self.current_frame.copy()

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # Copie temporaire pour dessiner
                img = self.temp_image.copy()
                cv2.rectangle(img, (self.start_x, self.start_y), (x, y), (0, 255, 0), 2)
                cv2.imshow(self.window_name, img)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            rectangle = (self.start_x, self.start_y, x, y)
            self.rectangles.append(rectangle)
            cv2.rectangle(self.current_frame, (self.start_x, self.start_y), (x, y), (0, 255, 0), 2)
            cv2.imshow(self.window_name, self.current_frame)

    def load_frame(self, frame_index):
        # Positionner le lecteur de vidéo à la frame demandée
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.capture.read()
        if ret:
            self.current_frame = frame
            self.rectangles = []  # Réinitialiser les rectangles pour la nouvelle image
        else:
            raise ValueError("Impossible de lire la frame")

    def show(self):
        while True:
            # Afficher l'image dans la fenêtre
            cv2.imshow(self.window_name, self.current_frame)

            key = cv2.waitKey(0)  # Attendre une touche

            if key == 27:  # Touche 'Esc' pour quitter
                break
            elif key == 13:  # Touche 'Entrée' pour passer à l'image suivante
                self.write_datas()
                self.next_image()
            elif key == ord('f'):  # Touche 'f' pour supprimer le dernier rectangle
                self.remove_last_rectangle()

        # Nettoyer
        self.capture.release()
        cv2.destroyAllWindows()
        
    def write_datas(self, frame):
        pass

    def next_image(self):
        # Passer à l'image suivante
        self.current_image_index += 1
        if self.current_image_index < len(self.frame_indices):
            self.load_frame(self.frame_indices[self.current_image_index])
        else:
            print("Aucune autre image disponible.")
            self.current_frame = np.ones((500, 500, 3), np.uint8) * 255

    def remove_last_rectangle(self):
        if self.rectangles:
            # Supprimer le dernier rectangle
            self.rectangles.pop()
            # Recharger l'image pour effacer les dessins précédents
            self.load_frame(self.frame_indices[self.current_image_index])
            # Redessiner les rectangles restants
            for rect in self.rectangles:
                x1, y1, x2, y2 = rect
                cv2.rectangle(self.current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow(self.window_name, self.current_frame)
        else:
            print("Aucun rectangle à supprimer.")

if __name__ == '__main__':
    video_path = '/Users/georgesschmidt/VisualCodeProjects/SafaryCount/videos/elephant.mp4'
    drawer = VideoRectangleDrawer(video_path, num_images=30)
    drawer.show()
