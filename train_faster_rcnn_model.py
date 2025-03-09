
import os
import torch
from typing import List
from torchvision import datasets 
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
 
from detection_model import DetectionModel
from helpers import Row, export_training_results_to_csv
from image_data import ImageData 

# csv_file="fasterrcnn_resnet50_results.csv"
# fasterrcnn_model_path = "models/faster_rcnn/fasterrcnn_resnet50.pth"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


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
    


# def get_fasterrcnn_model():
#     # Load the pre-trained Faster R-CNN model with a ResNet-50 backbone
#     model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
 
#     if os.path.exists(fasterrcnn_model_path):
#         model.load_state_dict(torch.load(fasterrcnn_model_path, weights_only=True))
    
#     return model



 


model_path = "models/faster_rcnn/fasterrcnn_resnet50.pth"
csv_results_path = "fasterrcnn_resnet50_results.csv"

class FasterRcnn(DetectionModel):
    def __init__(self):
        super().__init__(model_path, csv_results_path)
        self._model: FasterRCNN =  None

    @property
    def model(self): 
        if not self._model:
            # Load the pre-trained Faster R-CNN model with a ResNet-50 backbone
            self._model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
            in_features = self._model.roi_heads.box_predictor.cls_score.in_features
            self._model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
        
            if os.path.exists(self.model_path):
                self._model.load_state_dict(torch.load(self.model_path, weights_only=True))
            else: 
                self.train_model()

        return self._model

    def train_model(self):
        train_dataset: datasets.ImageFolder = AncientScrollDataset('saraay@post.jce.ac.il')
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
        
        # Define optimizer and learning rate scheduler
        optimizer = torch.optim.SGD(self._model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        
        # Training loop
        self._model.to(device)
        self._model.train()

        for epoch in range(0):
            self.train_one_epoch(optimizer, train_loader, device, epoch)

            # Save the model's state dictionary after every epoch
            torch.save(self._model.state_dict(), self._model)
        
        
 
    def train_one_epoch(self, optimizer, data_loader, device, epoch):
        for images, targets, _ in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backpropagation
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch [{epoch}] Loss: {losses.item()}")


    def evaluation(self, images_path: str) -> List[Row]:
        dataset: datasets.ImageFolder = AncientScrollDataset(images_path)
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











def main(train_loader, valid_loader):
    train_dataset: datasets.ImageFolder = AncientScrollDataset('saraay@post.jce.ac.il')
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    
    test_dataset: datasets.ImageFolder = AncientScrollDataset('saraay@post.jce.ac.il')
    valid_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # Load the pre-trained Faster R-CNN model with a ResNet-50 backbone
    fasterrcnn_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    in_features = fasterrcnn_model.roi_heads.box_predictor.cls_score.in_features
    fasterrcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
 
    if os.path.exists(fasterrcnn_model_path):
        fasterrcnn_model.load_state_dict(torch.load(fasterrcnn_model_path, weights_only=True))
    
    # Define optimizer and learning rate scheduler
    optimizer = torch.optim.SGD(fasterrcnn_model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # Training loop
    fasterrcnn_model.to(device)
    fasterrcnn_model.train()

    for epoch in range(0):
        train_one_epoch(fasterrcnn_model, optimizer, train_loader, device, epoch)

        # Save the model's state dictionary after every epoch
        torch.save(fasterrcnn_model.state_dict(), fasterrcnn_model_path)

    fasterrcnn_model.eval()
    
    final_analyze_train_result: List[Row] = []
    with torch.no_grad():
        for images, targets, images_data in valid_loader:
            images = list(img.to(device) for img in images)
            
            predictions = fasterrcnn_model(images)
            
            for i, prediction in enumerate(predictions):
                pred_bbox = prediction['boxes']
                image_data: ImageData = images_data[i]
                img_train_result: List[Row] = image_data.analyze_train_result(pred_bbox)
                final_analyze_train_result.extend(img_train_result)


    export_training_results_to_csv(
        csv_file=csv_file, 
        train_result=final_analyze_train_result
    )


if __name__ == "__main__":
    main()