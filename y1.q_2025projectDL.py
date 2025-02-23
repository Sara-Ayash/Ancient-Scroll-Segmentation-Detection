import json
from typing import List
from pydantic import BaseModel 
import os 
from PIL import Image
import torch
from torch.utils.data import DataLoader 
from torchvision.models.detection import fasterrcnn_resnet50_fpn  
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim




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
    def __init__(self, data_dir, include_transform=True):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.lower().endswith('.jpg')]
        self.bboxes_details_path = os.path.join(self.data_dir, 'bboxes_details')
        self.transform = None if not include_transform else self.set_transform()
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_name = self.file_list[idx] 
        file_path = os.path.join(self.data_dir, file_name)
    
        # prepare image and target
        image = Image.open(file_path).convert("RGB")
        
        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)
        target = self.get_shapes(file_path)
        
        return image, target
    

    def set_transform(self):
        return transforms.Compose([
            transforms.ToTensor(), 
        ])
 
    def get_shapes(self, file_path):
        bboxes, labels, shapes  = [], [], []
        bboxes_json_path = os.path.join(self.bboxes_details_path, os.path.splitext(os.path.basename(file_path))[0] + ".json")
        
        with open(bboxes_json_path) as f:
            shapes =  json.load(f).get("shapes", [])
        
        # return [
        #     Shape(label=shape.get('label'), bbox=sum(shape.get('points', []), []))
        #     for shape in shapes
        # ]
        for shape in shapes:
            labels.append(int(shape.get('label')))
            bboxes.append(sum(shape.get('points', []), []))

        return {
            "boxes": torch.tensor(bboxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.float32),
        } 

    
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    train_loss = 0.0
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{
            k: v.to(device) 
            for k, v in t.items()} 
            for t in targets]

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        losses.backward()
        optimizer.step()
        train_loss += losses.item()

    print(f"Epoch [{epoch}] Loss: {losses.item():.4f}")        
           
        
        
def get_fasterrcnn_model():
    # Load the pre-trained Faster R-CNN model with a ResNet-50 backbone
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the head of the model with a new one (for the number of classes in your dataset)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 1)

    return model

class Model(BaseModel):
    model_path: str
    lr: float = 0.005
    
    
     
if __name__ == "__main__":
    train_dataset: datasets.ImageFolder = AncientScrollDataset('saraay@post.jce.ac.il/training_data')
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    # validate_dataset: datasets.ImageFolder = AncientScrollDataset('student_318411840_v2/Validation/Validation/smoking')
    
    # train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    # validation_loader = DataLoader(validate_dataset, batch_size=2, shuffle=True, num_workers=4)
    
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    
    # ---------------- Faster R-CNN with ResNet-50 ----------------
    fasterrcnn_model_path = "fasterrcnn_resnet50.pth"
    model = get_fasterrcnn_model()
    
    if os.path.exists(fasterrcnn_model_path):
        model.load_state_dict(torch.load(fasterrcnn_model_path, weights_only=True))
        
    model.to(device)
    
    # Define optimizer and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        lr_scheduler.step()
        
        # Save the model's state dictionary after every epoch
        torch.save(model.state_dict(), fasterrcnn_model_path)
        print(f"Model saved: {fasterrcnn_model_path}")
 
    
   