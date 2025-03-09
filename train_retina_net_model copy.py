
import os
import torch
from typing import List
from helpers import Row
from image_data import ImageData 
from torchvision import datasets 
from torch.utils.data import DataLoader
from detection_model import DetectionModel 
from PIL import Image
import json
from effdet import get_efficientdet_config, EfficientDet
from effdet.data import create_dataset
import torch.nn as nn
from torchvision import transforms

from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_path = "models/retinanet/retinanet_resnet50.pth"
csv_results_path = "retinanet_resnet50_results.csv" 


transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class RetinaNetDataset(datasets.VisionDataset):
    def __init__(self, data_dir, if_train=True):
        self.file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith('.jpg')]
        self.if_train = if_train

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        image_data = ImageData(file_name,transform=transform)

        image = image_data.get_image()
        boxes = image_data.get_bboxes()
        labels = image_data.get_labels()

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32), 
            'labels': torch.tensor(labels, dtype=torch.int64)
        }

        return image, target, image_data
    

class RetinaNetModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(RetinaNetModel, self).__init__()
        backbone = resnet_fpn_backbone('resnet50', pretrained=pretrained)
     
        self.model = RetinaNet(
            backbone,
            num_classes
        )
    
    def forward(self, images, targets=None):
        return self.model(images, targets)
    
class RetinaNetDetectionModel(DetectionModel):
    def __init__(self):
        super().__init__(model_path, csv_results_path)
        self._model: RetinaNetModel =  None

    @property
    def model(self): 
        if not self._model:
            self._model = RetinaNetModel(num_classes=2).model
             
            if os.path.exists(self.model_path):
                self._model.load_state_dict(torch.load(self.model_path, weights_only=True))
            else: 
                print("1", id(self._model))
                self.train_model(self._model)
        
        print("5", id(self._model))
        return self._model

    def train_model(self, model_to_train: RetinaNetModel):
        train_dataset: datasets.ImageFolder = RetinaNetDataset('saraay@post.jce.ac.il')
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
        
        print("2", id(self._model))
        # Define optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(model_to_train.parameters(), lr=0.0001)
        
        # Training loop
        model_to_train.to(device)
        model_to_train.train()

        for epoch in range(5):
            self.train_one_epoch(model_to_train, optimizer, train_loader, device, epoch)

            # Save the model's state dictionary after every epoch
            torch.save(model_to_train.state_dict(), self.model_path)
        
        
 
    def train_one_epoch(self, model_to_train: RetinaNetModel, optimizer, data_loader, device, epoch):
        print("3", id(model_to_train))
        for images, targets, _ in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            loss_dict = model_to_train(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backpropagation
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch [{epoch}] Loss: {losses.item()}")


    def evaluation(self, image_path) -> List[Row]:
        image_data = ImageData(image_path)
         
        self.model.eval()
        img = image_data.get_image().to(device)
        predictions = self.model([img])

        pred_bbox = predictions[0]['boxes']
        img_train_result: List[Row] = image_data.analyze_test_result(pred_bbox)
        return img_train_result

        

    def validation_dataset(self, image_path: str) -> List[Row]:
        dataset: datasets.ImageFolder = RetinaNetDataset(image_path)
        data_loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

        self.model.eval()
        
        final_analyze_train_result: List[Row] = []
        with torch.no_grad():
            for images, _, images_data in data_loader:
                images = list(img.to(device) for img in images)
                
                predictions = self.model(images)
                
                for i, prediction in enumerate(predictions):
                    pred_bbox = prediction['boxes']
                    image_data: ImageData = images_data[i]
                    img_train_result: List[Row] = image_data.analyze_train_result(pred_bbox)
                    final_analyze_train_result.extend(img_train_result)

        return final_analyze_train_result



if __name__ == "__main__":
    retinanet_model = RetinaNetDetectionModel()
    modelush = retinanet_model.model 
    res = modelush.validation_dataset('saraay@post.jce.ac.il')
    pass