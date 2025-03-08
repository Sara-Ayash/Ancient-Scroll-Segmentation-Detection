import csv
from typing import List
import numpy as np
from pydantic import BaseModel
import torch

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



def train_one_epoch(model, optimizer, data_loader, device, epoch):
    for images, targets, _ in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backpropagation
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch [{epoch}] Loss: {losses.item()}")