import os
import cv2
import torch
from typing import List
from ultralytics import YOLO
from image_data import ImageData
from helpers import Row, export_training_results_to_csv

 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
yolo_model_path = 'runs/detect/train/weights/best.pt'
csv_file = 'models/yolo/yolov8n_results.csv'

def main():
    yolo_model_path = 'runs/detect/train/weights/best.pt'
    
    if os.path.exists(yolo_model_path):
        model = YOLO(yolo_model_path) 
    else:
        model = YOLO("yolov8n.pt") 
        model.train(data="models/yolo/datasets/ancient_scroll_dataset/dataset.yaml", epochs=200)
        model.val()

    image_path = "saraay@post.jce.ac.il/M40979-1-E.jpg"
    image_data: ImageData = ImageData(image_path)
    image = cv2.imread(image_path)

    # Run inference
    results = model(image, conf=0.1)
    final_analyze_train_result: List[Row] = []
    # Extract bounding boxes and other details
    for result in results:
        pred_bboxes = result.boxes.xyxy
        img_train_result: List[Row] = image_data.analyze_train_result(pred_bboxes)
        final_analyze_train_result.extend(img_train_result)
 
    export_training_results_to_csv(
        csv_file=csv_file, 
        train_result=final_analyze_train_result
    )
 
    
def prepare_data(dir_path):
    images_list = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith('.jpg')]

    class_id = 0
    for image_path in images_list:
        image_data = ImageData(image_path=image_path)
        bboxes = image_data.get_bboxes(format="yolo")
        yolo_bboxes = []
        for bbox in bboxes:
            x_center, y_center, width, height = bbox
            yolo_bboxes.append((class_id, x_center, y_center, width, height))
        
        labels_dir = "models/yolo/datasets/ancient_scroll_dataset/labels/train"
        labels_path = os.path.join(labels_dir, f'{image_data.image_name}.txt')
        with open(labels_path, "w") as f:
            for bbox in yolo_bboxes:
                f.write(" ".join([str(x.item()) if isinstance(x, torch.Tensor) else str(x) for x in bbox]) + "\n")

if __name__ == "__main__":

    # prepare the data
    # prepare_data('saraay@post.jce.ac.il')

    main()