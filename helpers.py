import csv
import os
import cv2
import torch
import numpy as np
import pandas as pd
from typing import List
from pydantic import BaseModel


class Row(BaseModel):
    image_name: str
    scroll_number: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    iou: float = -1
    

def correct_bounding_boxes(bboxes):
    # Ensure input is a NumPy array
    bboxes = np.asarray(bboxes)
    
    # Correct x coordinates
    # Swap the x coordinates if the top-left x is greater than the bottom-right x
    x_min = np.minimum(bboxes[:, 0], bboxes[:, 2])
    x_max = np.maximum(bboxes[:, 0], bboxes[:, 2])
    
    # Correct y coordinates
    # Swap the y coordinates if the top-left y is greater than the bottom-right y
    y_min = np.minimum(bboxes[:, 1], bboxes[:, 3])
    y_max = np.maximum(bboxes[:, 1], bboxes[:, 3])
    
    # Construct the corrected bounding boxes array
    corrected_bboxes = np.stack([x_min, y_min, x_max, y_max], axis=1)
    
    return corrected_bboxes


def sort_by_distance(bbox_list): 
    # Compute distances from (0,0) using the top-left point (x_min, y_min)
    distances = torch.sqrt(bbox_list[:, 0]**2 + bbox_list[:, 1]**2)

    # Sort indices based on distance
    sorted_indices = torch.argsort(distances)

    # Sort the tensor
    sorted_tensor = bbox_list[sorted_indices]

    return sorted_tensor


def export_training_results_to_csv(train_result: List[Row], csv_file):
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=train_result[0].model_dump().keys())
        writer.writeheader()   
        for res in train_result:
            writer.writerow(res.model_dump())  
    
    print(f"CSV file '{csv_file}' has been created successfully.")



def draw_bounding_boxes_from_csv(image_dir_path, csv_path):
    output_folder = "output_images/"
    os.makedirs(output_folder, exist_ok=True)

    image_name = None
    
    df = pd.read_csv(csv_path)

    grouped = df.groupby('image_name')
    
    for image_name, group in grouped:  
        image_name += ".jpg"
        image_path = os.path.join(image_dir_path, image_name)
        image = cv2.imread(image_path)
    
        for _, row in group.iterrows():
            xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            confidence = row.get('iou', -1)
            label = row.get('scroll_number', "scroll")

            # Draw rectangle
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

            # Draw label and confidence
            text = f"{label}: {confidence:.2f}"
            cv2.putText(image, text, (xmin, max(ymin - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)  # Bigger font
        
        output_path = os.path.join(output_folder, image_name)    
        cv2.imwrite(output_path, image)
        print(f"Saved: {output_path}")




# def draw_bounding_boxes_from_csv(image_path, csv_path):
#     output_folder = "output_images/"
#     os.makedirs(output_folder, exist_ok=True)

#     image_name: str = os.path.splitext(os.path.basename(image_path))[0]
    
#     df = pd.read_csv(csv_path)
#     image = cv2.imread(image_path)

#     for _, row in df.iterrows():
#         xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
#         confidence = row.get('iou', -1)
#         label = row.get('scroll_number', "scroll")

#         # Draw rectangle
#         cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

#         # Draw label and confidence
#         text = f"{label}: {confidence:.2f}"
#         cv2.putText(image, text, (xmin, max(ymin - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)  # Bigger font
    
#     image_name += ".JPG"
#     output_path = os.path.join(output_folder, image_name)    
#     cv2.imwrite(output_path, image)
#     print(f"Saved: {output_path}")