
import os
import json
import numpy as np
import torch
from typing import List
from PIL import Image
from torchvision import datasets, transforms
from pydantic import BaseModel
from torchvision.tv_tensors import BoundingBoxes

from helpers import Row, correct_bounding_boxes, sort_by_distance



class ImageData():
    def __init__(self, image_path, include_transform=True):
        self.image_path: str = image_path
        self.image_name: str = os.path.splitext(os.path.basename(image_path))[0]
        self.annotation_path = os.path.join('saraay@post.jce.ac.il', self.image_name + ".json")
        self.transform = None if not include_transform else self.set_transform()
        self._bboxes = torch.tensor([])
        self.rows: List[Row] = [] 

    
    def set_transform(self):
        return transforms.Compose([
            transforms.ToTensor(), 
        ])
    
    def get_bboxes(self):
        if not self._bboxes.any():
            with open(self.annotation_path) as f:
                shapes =  json.load(f).get("shapes", [])

            # Process bounding box annotations
            points_list = np.array([shape['points'] for shape in shapes]).reshape(len(shapes), 4)
            bbox_list = correct_bounding_boxes(points_list)
            self._bboxes = torch.Tensor(bbox_list)
        return self._bboxes

    def get_labels(self):
        with open(self.annotation_path) as f:
            shapes =  json.load(f).get("shapes", [])
        scrolls = [shape['label'] for shape in shapes]
        return torch.Tensor([1 for _ in scrolls])


    def get_image(self):
        image = Image.open(self.image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


    def calculate_iou(self, box_pred, box_gt):
        x_left = max(box_pred[0], box_gt[0])
        y_top = max(box_pred[1], box_gt[1])
        x_right = min(box_pred[2], box_gt[2])
        y_bottom = min(box_pred[3], box_gt[3])

        intersection_width = max(0, x_right - x_left)
        intersection_height = max(0, y_bottom - y_top)
        area_intersection = intersection_width * intersection_height

        area_pred = (box_pred[2] - box_pred[0]) * (box_pred[3] - box_pred[1])
        area_gt = (box_gt[2] - box_gt[0]) * (box_gt[3] - box_gt[1])


        area_union = area_pred + area_gt - area_intersection

        iou = area_intersection / area_union if area_union > 0 else 0
        return iou
    
    def get_max_iou(self, bbox, sorted_pred_bboxes):
        iou_per_pred_bbox = {
            pred_bbox: self.calculate_iou(bbox, pred_bbox)  
            for pred_bbox in sorted_pred_bboxes 
        }

        return max(iou_per_pred_bbox.items(), key=lambda x: x[1])
      

    def analyze_train_result(self, pred_bboxes):
        bboxes = self.get_bboxes()
        sorted_bboxes = sort_by_distance(bboxes)
        
        img_train_result: List[Row] = []

        for i, bbox in enumerate(sorted_bboxes):
            scroll_number = i+1
            match_pred_bbox, iou = self.get_max_iou(bbox, pred_bboxes)
            xmin, ymin, xmax, ymax = match_pred_bbox
            row = Row(
                image_name=self.image_name,
                scroll_number=scroll_number,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                iou=iou
            )
            img_train_result.append(row)

        return img_train_result
 
    def analyze_test_result(self, pred_bboxes):
        sorted_bboxes = sort_by_distance(pred_bboxes)
        img_train_result = []
        
        for i, bbox in enumerate(sorted_bboxes):
            scroll_number = i+1
            xmin, ymin, xmax, ymax = bbox
            row = Row(
                image_name=self.image_name,
                scroll_number=scroll_number,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax
            )
            img_train_result.append(row)

        return img_train_result




        # boxes = BoundingBoxes(bbox_tensor, format='xyxy', canvas_size=image.size[::-1])
 
 

class AncientScrollDataset(datasets.VisionDataset):
    def __init__(self, data_dir, if_train=True):
        self.file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith('.jpg')]
        self.if_train = if_train


    def __len__(self):
        return len(self.file_list)


    def __getitem__(self, idx):
        file_name = self.file_list[idx] 
        image_data = ImageData(file_name)

        image = image_data.get_image()
        boxes = image_data.get_bboxes()
        labels = image_data.get_labels()

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32), 
            'labels': torch.tensor(labels, dtype=torch.int64)
        }

        return image, target, image_data
    