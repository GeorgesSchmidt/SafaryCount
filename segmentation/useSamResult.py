import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from imgbeddings import imgbeddings
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

class DetectElephant:
    def __init__(self, path_img, path_img_sam, model, n_clusters=3, n_neighbors=3) -> None:
        self.ibed = imgbeddings()
        self.img = cv2.imread(path_img)
        self.img_sam = cv2.imread(path_img_sam, cv2.IMREAD_GRAYSCALE)
        self.model = YOLO(model)
        print('names', self.model.names)
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.get_prediction()
        self.create_model()
        self.get_boxes_sam()
        self.compare_embeddings()
        
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
        
        print('emb yolo', len(self.embed_yolo))
        cv2.imshow('', image)
        cv2.waitKey(0)
        
    def create_model(self):
        if len(self.embed_yolo) == 0:
            print("No embeddings found to create model.")
            return
        
        # Clustering with KMeans
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.clusters = kmeans.fit_predict(self.embed_yolo)
        self.cluster_centers = kmeans.cluster_centers_
        
        # KNN Classifier
        self.knn_model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.knn_model.fit(self.cluster_centers, np.arange(self.n_clusters))
        
        print('Clusters:', self.n_clusters)
        
    def get_boxes_sam(self):
        _, bin = cv2.threshold(self.img_sam, 0, 255, cv2.THRESH_BINARY)
        bin = cv2.bitwise_not(bin)
        contours, _ = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        color = self.img.copy()
        self.emb_sam = []
        self.rois = []
        for contour in contours:
            box = cv2.boundingRect(contour)
            x, y, w, h = box
            self.rois.append(box)
            img = self.img[y:y+h, x:x+w]
            emb = self.get_embeddings(img)[0]
            self.emb_sam.append(emb)
            cv2.rectangle(color, (x, y), (x+w, y+h), (0, 255, 0), 2)
        print('emb sam', len(self.emb_sam))
        
    def get_embeddings(self, image):
        pil_img = Image.fromarray(image)
        pil_img = pil_img.convert("RGB")
        return self.ibed.to_embeddings(pil_img)
        
    def compare_embeddings(self):
        image = self.img.copy()
        for i, emb_sam in enumerate(self.emb_sam):
            # Find the nearest cluster center
            label = self.knn_model.predict([emb_sam])[0]
            cluster_center = self.cluster_centers[label]
            
            # Calculate similarity (cosine distance or Euclidean distance)
            distance = np.linalg.norm(emb_sam - cluster_center)
            
            
            # Define a threshold for detection
            threshold = 14  # Adjust based on validation
            if distance < threshold:
                print(f'Distance for SAM embedding {i}: {distance}')
                print(f'Elephant detected for SAM embedding {i}')
                x, y, w, h = self.rois[i]
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                print(f'No elephant detected for SAM embedding {i}')
        cv2.imshow('', image)
        cv2.waitKey(0)

if __name__ == '__main__':
    path_img = '/Users/georgesschmidt/VisualCodeProjects/SafaryCount/pictures/elephant_lake.png'
    path_img_sam = '/Users/georgesschmidt/VisualCodeProjects/SafaryCount/pictures/elephant_lake_result_sam.png'
    model = '/Users/georgesschmidt/VisualCodeProjects/SafaryCount/weights/yolo_equitable_100.pt'
    DetectElephant(path_img, path_img_sam, model, n_clusters=3, n_neighbors=3)
