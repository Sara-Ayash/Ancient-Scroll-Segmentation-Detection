import json
from typing import List
from pydantic import BaseModel, Field
# #!/usr/bin/env python3
# import time
# import os 
# from PIL import Image
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
from torchmetrics.detection import IntersectionOverUnion



def predict_process_bounding_boxes(image_path: str, output_csv: str) -> None:
    """
    Processes an image to detect bounding boxes around scroll segments.
    Saves the bounding box data to a CSV file.
    Args:
    image_path (str): Path to the input image.
    output_csv (str): Path to the output CSV file.
    """
pass




        
image_name = "M40587-1-C"
final_cvs_path = "y1.q_2025projectDL_csv.csv"

class Points(BaseModel):
    min_point: List[float]
    max_point: List[float]
    
    def area(self):
        xmin, xmax = self.min_point
        ymin, ymax = self.max_point
        return (xmax-xmin) * (ymin-ymax)

class Shape(BaseModel):
    label: int
    bbox: List[float]


class BoundingBox(BaseModel):
    image_name: str
    shapes: List[Shape]
    iou: int = -1

     
class Shape(BaseModel):
    label: int
    bbox: List[float]


    
    
def IOU(bbox, result_bbox):
    xmin_a, xmax_a, ymin_a, ymax_a = bbox
    xmin_b, xmax_b, ymin_b, ymax_b = result_bbox
    point_a = Points(min_point=[xmin_a, xmax_a], max=[ymin_a, ymax_a])
    point_b = Points(min_point=[xmin_b, xmax_b], max=[ymin_b, ymax_b])

    area_inter = if_intersection(xmin_a, xmax_a, ymin_a, ymax_a, xmin_b, xmax_b, ymin_b, ymax_b)
    if area_inter:
        area_a = point_a.area
        area_b = point_a.area
        iou = float(area_inter) / (area_a + area_b - area_inter)
        return iou
    return 0

def if_intersection(xmin_a, xmax_a, ymin_a, ymax_a, xmin_b, xmax_b, ymin_b, ymax_b) -> float:
    if_intersect = False
    if xmin_a < xmax_b <= xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_a <= xmin_b < xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_b < xmax_a <= xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    elif xmin_b <= xmin_a < xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    else:
        return if_intersect
    if if_intersect:
        x_sorted_list = sorted([xmin_a, xmax_a, xmin_b, xmax_b])
        y_sorted_list = sorted([ymin_a, ymax_a, ymin_b, ymax_b])
        x_intersect_w = x_sorted_list[2] - x_sorted_list[1]
        y_intersect_h = y_sorted_list[2] - y_sorted_list[1]
        area_inter = x_intersect_w * y_intersect_h
        return area_inter
    return 0
  

class AncientScrollDataset(datasets.VisionDataset):
    def __init__(self, data_dir ):
        self.data_dir = data_dir
        self.transform = None if not include_transform else self.set_transform()
        self.file_list = os.listdir(data_dir)
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_name = self.file_list[idx].rsplit('.')
        file_path = os.path.join(self.data_dir, file_name)
        
        # prepare image and target
        image = self.prepare_image(file_path)
        target = self.get_shapes(file_path)
        
        return image, target
    
    def prepare_image(self, image_path):
        image = Image.open(image_path).convert("RGB") 
        image_tensor = F.to_tensor(image).unsqueeze(0)  
        return image_tensor.to(device)
    
    def get_shapes(self, file_path):
        shapes = []
        
        with open(file_path) as f:
            shapes =  json.load(f).get("shapes", [])
        
        return [
            Shape(label=shape.get('label'), bbox=sum(shape.get('points', []), []))
            for shape in shapes
        ]
        
        # for shape in shapes:
        #     scroll_number.append(shape.get('label'))
        #     bboxes.append(sum(shape.get('points', []), []))

        # return {
        #     "boxes": torch.tensor(boxes, dtype=torch.float32).to(device),
        #     "scroll_number": torch.tensor(scroll_number, dtype=torch.int64).to(device),
        # } 

    
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    for images, targets in data_loader:
        # extract targets
        processed_targets, valid_images = [], []
        for target in enumerate(targets):
            boxes, labels = [], []
            
            for obj in target: 
                bbox = obj["bbox"]     
                bboxes_gt.append(bbox)
                labels.append(obj["label"])

            if boxes:
                processed_target = {
                    "boxes": torch.tensor(boxes, dtype=torch.float32).to(device),
                    "labels": torch.tensor(labels, dtype=torch.int64).to(device),
                }
                processed_targets.append(processed_target)
        
        images = images.to(device)
        
        # Forward pass
        outputs = model(images).squeeze()
         
        loss_dict = model(images, processed_targets)
        bboxes_pred = outputs["instances"].pred_boxes
        IOUs = metric.update(bboxes_pred, processed_targets)
        # losses = sum(loss for loss in loss_dict.values())

        # Backpropagation
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch [{epoch}] Loss: {losses.item():.4f}")        
           
        
        
def get_fasterrcnn_model():
    # Load pre-trained Faster R-CNN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 1)
    return model

class Model(BaseModel):
    model_path: str
    lr: float = 0.005
    
    
     
if __name__ == "__main__":
    train_dataset: datasets.ImageFolder = AncientScrollDataset('saraay@post.jce.ac.il/training_data')
    validate_dataset: datasets.ImageFolder = AncientScrollDataset('student_318411840_v2/Validation/Validation/smoking')
    
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validate_dataset, batch_size=2, shuffle=True, num_workers=4)
    
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    
    # ---------------- Faster R-CNN with ResNet-50 ----------------
    fasterrcnn_model_path = "fasterrcnn_resnet50.pth"
    model = get_fasterrcnn_model()
    
    if os.path.exists(fasterrcnn_model_path):
        model.load_state_dict(torch.load(fasterrcnn_model_path, weights_only=True))
        
    model.to(device)
    
    # Define optimizer and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        lr_scheduler.step()
        
        # Save the model's state dictionary after every epoch
        torch.save(model.state_dict(), model_path)
        print(f"Model saved: {model_path}")
 
    
   