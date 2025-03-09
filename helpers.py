import csv
import os
import cv2
import torch
import numpy as np
import pandas as pd
from typing import List
from pydantic import BaseModel
import matplotlib.pyplot as plt


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



def draw_frames(image_path, csv_output):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    df = pd.read_csv(csv_output)
    # Assuming CSV has columns: x_min, y_min, x_max, y_max, confidence, label
    for _, row in df.iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence = row.get('iou', -1)  # Default to 1.0 if confidence column is missing
        label = row.get('scroll_number', "scroll")  # Default label if missing

        # Draw rectangle
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Put label and confidence score
        text = f"{label}: {confidence:.2f}"
        cv2.putText(image, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 2)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image_rgb)
    plt.axis("off")  # Hide axes
    plt.show()