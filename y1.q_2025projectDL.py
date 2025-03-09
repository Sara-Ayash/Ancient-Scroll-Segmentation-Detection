import torch
from helpers import draw_frames, export_training_results_to_csv
from train_faster_rcnn_model import FasterRcnnModel
from train_retina_net_model import RetinaNetDetectionModel
from train_yolo_model import YoloV8Model  
 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_best_model(best_model= None):
    if best_model == "yolov8": 
        return YoloV8Model()
    elif best_model == "faster_rcnn":
        return FasterRcnnModel()
    else:
        return RetinaNetDetectionModel()
    
    
def predict_process_bounding_boxes(image_path: str, output_csv: str) -> None:
    """
    Processes an image to detect bounding boxes around scroll segments.
    Saves the bounding box data to a CSV file.
    Args:
    image_path (str): Path to the input image.
    output_csv (str): Path to the output CSV file.
    """ 

    detection_model = get_best_model()
    detection_model.csv_results_path = output_csv

    img_train_result = detection_model.evaluation(image_path)

    export_training_results_to_csv(
        csv_file=detection_model.csv_results_path, 
        train_result=img_train_result
    )

    return detection_model.csv_results_path

if __name__ == "__main__":
    image_path = 'saraay@post.jce.ac.il/test/M40588-1-C.jpg'
    csv_file = 'predict_bboxes_M40588.csv'
    csv_results_path = predict_process_bounding_boxes(image_path, csv_file)
    draw_frames(image_path, csv_file)
 
