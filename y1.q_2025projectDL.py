import torch
from typing import List
from helpers import Row, export_training_results_to_csv
from image_data import ImageData
import train_faster_rcnn_model as faster_rcnn
import train_yolo_model as yolov8_model
 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_best_model(best_model):
    if best_model == "yolov8": 
        return yolov8_model.YoloV8() 
    elif best_model == "faster_rcnn":
        model = faster_rcnn.get_fasterrcnn_model()
        model.eval()
        return model

    
def predict_process_bounding_boxes(image_path: str, output_csv: str) -> None:
    """
    Processes an image to detect bounding boxes around scroll segments.
    Saves the bounding box data to a CSV file.
    Args:
    image_path (str): Path to the input image.
    output_csv (str): Path to the output CSV file.
    """ 

    detection_model = get_best_model("yolov8")
    
    img_train_result = detection_model.evaluation(image_path)

    export_training_results_to_csv(
        csv_file=output_csv, 
        train_result=img_train_result
    )

if __name__ == "__main__":
    image_path = 'saraay@post.jce.ac.il/M40587-1-C.jpg'
    csv_file = 'predict_bboxes.csv'
    predict_process_bounding_boxes(image_path, csv_file)
 