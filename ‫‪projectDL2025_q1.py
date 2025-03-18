import os
import torch
from helpers import draw_bounding_boxes_from_csv, export_training_results_to_csv
from train_faster_rcnn_model import FasterRcnnModel
from train_yolo_model import YoloV8Model  
 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_best_model(best_model= None):
    if best_model == "yolov8": 
        return YoloV8Model()
    else: return FasterRcnnModel()
    
    
def predict_process_bounding_boxes(image_path: str, output_csv: str) -> None:
    """
    Processes an image to detect bounding boxes around scroll segments.
    Saves the bounding box data to a CSV file.
    Args:
    image_path (str): Path to the input image.
    output_csv (str): Path to the output CSV file.
    """ 

    detection_model = get_best_model()
    # detection_model = get_best_model("yolov8")

    detection_model.csv_results_path = output_csv

    img_train_result = detection_model.evaluation(image_path)

    export_training_results_to_csv(
        csv_file=detection_model.csv_results_path, 
        train_result=img_train_result
    )


if __name__ == "__main__":
    validation_dir = "saraay@post.jce.ac.il/test"
    images = [os.path.join(validation_dir, f) for f in os.listdir(validation_dir) if f.lower().endswith('.jpg')]
    for image_path in images:
        csv_file = f'faster_rcnn_test_{os.path.basename(image_path)}.csv'
        csv_results_path = predict_process_bounding_boxes(image_path, csv_file)
        draw_bounding_boxes_from_csv(validation_dir, csv_file)
    
