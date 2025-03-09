import os
import cv2
import torch
from typing import List
from helpers import Row
from ultralytics import YOLO
from detection_model import DetectionModel
from image_data import ImageData

 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_path = 'runs/detect/train/weights/best.pt'
csv_results_path = 'models/yolo/yolov8n_results.csv'

class YoloV8Model(DetectionModel):
    
    def __init__(self):
        super().__init__(model_path, csv_results_path)
        self._model:YOLO = None
    
    @property        
    def model(self) -> YOLO:
        if not self._model:
            if os.path.exists(self.model_path):
                self._model = YOLO(self.model_path) 
            else:
                self._model = YOLO("yolov8n.pt") 
                self.train_model()
            
        return self._model
    
    def train_model(self):
        try:
            self._model.train(data="models/yolo/datasets/ancient_scroll_dataset/dataset.yaml", epochs=200)
            self._model.val()
        except Exception as error:
            print(f"Failed to train the YOLOv8 Modle, Error: {error}")

    def evaluation(self, image_path) -> List[Row]:
        image_data: ImageData = ImageData(image_path)
        image = cv2.imread(image_path)

        results = self.model(image, conf=0.1)
        final_analyze_train_result: List[Row] = []
        
        for result in results:
            pred_bboxes = result.boxes.xyxy
            img_train_result: List[Row] = image_data.analyze_train_result(pred_bboxes)
            final_analyze_train_result.extend(img_train_result)

        return final_analyze_train_result
    