
import os
import json
import numpy as np
import torch
import cv2
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

    def get_image_size(self):
        img = cv2.imread(self.image_path)
        height, width, _ = img.shape  # shape returns (height, width, channels)
        return width, height

    def convert_yolo_to_bbox(x_center, y_center, width, height, img_width, img_height):
        """ xmin, ymin, xmax, ymax (in absolute pixel values)"""
        xmin = int((x_center - width / 2.0) * img_width)
        ymin = int((y_center - height / 2.0) * img_height)
        xmax = int((x_center + width / 2.0) * img_width)
        ymax = int((y_center + height / 2.0) * img_height)

        return xmin, ymin, xmax, ymax
    
    
    def convert_bbox_to_yolo(self, bboxes):
        """ return x_center, y_center, width, height (normalized to 0-1) """
        yolo_bboxes = []
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox
            img_width, img_height = self.get_image_size()
                
            x_center = (xmax - xmin) / 2.0 / img_width
            y_center = (ymin + ymax) / 2.0 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            yolo_bboxes.append((x_center, y_center, width, height))

        return yolo_bboxes


    def get_bboxes(self, format="fasterrcnn")-> torch.Tensor:
        if not self._bboxes.any():
            with open(self.annotation_path) as f:
                shapes =  json.load(f).get("shapes", [])

            # Process bounding box annotations
            points_list = np.array([shape['points'] for shape in shapes]).reshape(len(shapes), 4)
            bbox_list = correct_bounding_boxes(points_list)
            if format == "yolo":
                bbox_list = self.convert_bbox_to_yolo(bbox_list)

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

