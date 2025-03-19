import os
import torch
from helpers import Row, draw_bounding_boxes_from_csv, export_training_results_to_csv
from faster_rcnn_model import FasterRcnnModel
from yolo_model import YoloV8Model  
 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_best_model(best_model= None):
    if best_model == "yolov8": 
        return YoloV8Model()
    else: return FasterRcnnModel()
    
    
def process_detailed_bounding_boxes(image_paths: list[str], output_csv: str) -> list[str]:
    """
    Processes a list of image file paths to detect detailed bounding bboxes
    for both large and small scroll segments.
    Saves the bounding box data to a CSV file.
    
    Args:
        image_paths (list[str]): List of full paths to the input images.
        output_csv (str): Path to the output CSV file.
   
    Returns:
        list[str]: List of full paths (path + file) name of processed images.
    """
    pred_results: list[Row] = []
    processed_images = []
    
    for image_path in image_paths:
        try:
            # detection_model = YoloV8Model()
            detection_model = FasterRcnnModel()
            detection_model.csv_results_path = output_csv

            img_train_result = detection_model.evaluation(image_path)

            pred_results.extend(img_train_result)
            processed_images.append(image_path)

        except Exception as error:
            print(f'An error occurred while trying to detect the scroll object for image: {image_path}. ERROR: {error}')

    export_training_results_to_csv(
        csv_file=detection_model.csv_results_path, 
        train_result=pred_results
    )
    return processed_images


# if __name__ == "__main__":
#     validation_dir = "test"
#     image_paths = [os.path.join(validation_dir, f) for f in os.listdir(validation_dir) if f.lower().endswith('.jpg')][10:]
#     csv_output = 'pred_results_part2_res.csv'
#     process_detailed_bounding_boxes(image_paths, csv_output)
#     draw_bounding_boxes_from_csv(validation_dir, csv_output)
 
    
