
import torch
from typing import List
from ancient_scroll_dataset import ImageData, Row
from helpers import export_training_results_to_csv
import train_faster_rcnn_resnet50_fpn_model as faster_rcnn
 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def predict_process_bounding_boxes(image_path: str, output_csv: str) -> None:
    """
    Processes an image to detect bounding boxes around scroll segments.
    Saves the bounding box data to a CSV file.
    Args:
    image_path (str): Path to the input image.
    output_csv (str): Path to the output CSV file.
    """ 
    image_data = ImageData(image_path)
    model_path = 'models/fasterrcnn_resnet50.pth'
    model = faster_rcnn.get_fasterrcnn_model(model_path)
    model.eval()
    img = image_data.get_image().to(device)
    predictions = model([img])
    
    pred_bbox = predictions[0]['boxes']
    img_train_result: List[Row] = image_data.analyze_test_result(pred_bbox)


    export_training_results_to_csv(
        csv_file=output_csv, 
        train_result=img_train_result
    )

    
 
if __name__ == "__main__":
    image_path = 'saraay@post.jce.ac.il/M40587-1-C.jpg'
    csv_file = 'predict_bboxes.csv'
    predict_process_bounding_boxes(image_path, csv_file)


 



    

    



