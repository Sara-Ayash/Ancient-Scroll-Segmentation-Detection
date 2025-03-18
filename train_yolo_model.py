import os
import cv2
import torch
from typing import List
from helpers import Row, draw_bounding_boxes_from_csv, export_training_results_to_csv
from ultralytics import YOLO
from image_data import ImageData
from detection_model import DetectionModel
from torchvision import datasets 


 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_path = 'models/yolo/best.pt'
csv_results_path = 'models/yolo/YOLOv8_validation_results.csv'

class YoloV8Model(DetectionModel):
    def __init__(self):
        super().__init__(model_path, csv_results_path)
        self._model: YOLO = None
    
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
        pred_bboxes = results[0].boxes.xyxy
        img_train_result: List[Row] = image_data.analyze_test_result(pred_bboxes)

        return img_train_result
    

    def validation_dataset(self, validation_dir: str) -> List[Row]:
        images = [os.path.join(validation_dir, f) for f in os.listdir(validation_dir) if f.lower().endswith('.jpg')]
        
        final_analyze_train_result: List[Row] = []
        
        for image_path in images:
            image_data: ImageData = ImageData(image_path)
            image = cv2.imread(image_path)

            results = self.model(image, conf=0.1)

            pred_bboxes = results[0].boxes.xyxy
            img_train_result: List[Row] = image_data.analyze_train_result(pred_bboxes)
            final_analyze_train_result.extend(img_train_result)

        return final_analyze_train_result

if __name__ == "__main__":
    modelush = YoloV8Model()
    validation_imgs_dir = 'saraay@post.jce.ac.il/validate'
    results = modelush.validation_dataset(validation_imgs_dir)
    csv_file = modelush.csv_results_path
    export_training_results_to_csv(
        csv_file=csv_file, 
        train_result=results
    )
    draw_bounding_boxes_from_csv(validation_imgs_dir, csv_file)


    
    
