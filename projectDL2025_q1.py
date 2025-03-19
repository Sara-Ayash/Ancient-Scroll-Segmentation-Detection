import os
import torch
from helpers import draw_bounding_boxes_from_csv, export_training_results_to_csv
from faster_rcnn_model import FasterRcnnModel
from yolo_model import YoloV8Model  
 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

   
def predict_process_bounding_boxes(image_path: str, output_csv: str) -> None:
    """
    Processes an image to detect bounding boxes around scroll segments.
    Saves the bounding box data to a CSV file.
    Args:
    image_path (str): Path to the input image.
    output_csv (str): Path to the output CSV file.
    """ 

    detection_model = YoloV8Model()
    # detection_model = FasterRcnnModel()

    detection_model.csv_results_path = output_csv

    img_train_result = detection_model.evaluation(image_path)

    export_training_results_to_csv(
        csv_file=detection_model.csv_results_path, 
        train_result=img_train_result
    )

# if __name__ == "__main__":
#     image_path = 'test/M40613-1-E.jpg'
#     csv_file = f'predection_results_{os.path.basename(image_path)}.csv'
#     csv_results_path = predict_process_bounding_boxes(image_path, csv_file)
#     draw_bounding_boxes_from_csv('test', csv_file)
    
