
import os
import torch
from typing import List
from helpers import Row
from image_data import ImageData 
from torchvision import datasets 
import torch.optim as optim
from torch.utils.data import DataLoader
from detection_model import DetectionModel
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet, RetinaNet_ResNet50_FPN_Weights



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_path = "models/retinanet/retinanet_resnet50.pth"
csv_results_path = "retinanet_resnet50_results.csv"

class CustomRetinaNet(RetinaNet):
    def __init__(self, num_classes=2):  # num_classes includes background
        # Use a pre-trained ResNet-50 backbone with FPN (Feature Pyramid Network)
        backbone = resnet_fpn_backbone('resnet50', pretrained=True)
        super().__init__(backbone, num_classes=num_classes)

    def forward(self, images, targets=None):
        # Pass images through the model
        return super().forward(images, targets)
    

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
    
def collate_fn(batch):
    images, targets, metadata = zip(*batch)
    images = torch.stack(images, dim=0)
    # targets can be a list of dictionaries, make sure it's processed properly
    return images, targets, metadata

class RetinanetModel(DetectionModel):
    def __init__(self):
        super().__init__(model_path, csv_results_path)
        self._model: RetinaNet=  None

    @property
    def model(self): 
        if not self._model:
            # Load the pre-trained Faster R-CNN model with a ResNet-50 backbone
            self._model = CustomRetinaNet(num_classes=2)

            if os.path.exists(self.model_path):
                self._model.load_state_dict(torch.load(self.model_path, weights_only=True))
            else: 
                print("1", id(self._model))
                self.train_model(self._model)
        
        print("5", id(self._model))
        return self._model

    def train_model(self, model_to_train: RetinaNet):

        
        train_dataset: datasets.ImageFolder = AncientScrollDataset('saraay@post.jce.ac.il')
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
        
        # Define the optimizer
        params = [p for p in model_to_train.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params, lr=1e-4, weight_decay=1e-4)

        # Learning rate scheduler (optional)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        # Training loop
        model_to_train.to(device)
        model_to_train.train()

        for epoch in range(5):
            self.train_one_epoch(model_to_train, optimizer, train_loader, device, epoch)

            # Save the model's state dictionary after every epoch
            torch.save(model_to_train.state_dict(), self.model_path)
            lr_scheduler.step()
        
 
    def train_one_epoch(self, model_to_train: RetinaNet, optimizer, data_loader, device, epoch):
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
        dataset: datasets.ImageFolder = AncientScrollDataset(image_path)
        data_loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
        model = self.model
        model.eval()
        
        final_analyze_train_result: List[Row] = []
        with torch.no_grad():
            for images, _, images_data in data_loader:
                images = [img.to(device) for img in images]

                predictions = model(images)  # Make predictions

                for i, prediction in enumerate(predictions):
                    # Get bounding boxes
                    pred_bbox = prediction['boxes']
                    pred_scores = prediction['scores']
                    
                    # Filter predictions based on confidence threshold
                    threshold = 0.5
                    filtered_bbox = pred_bbox[pred_scores > threshold]

                    # Access the metadata for the image
                    image_data = images_data[i]

                    # Process and analyze results
                    img_train_result = image_data.analyze_train_result(filtered_bbox)
                    final_analyze_train_result.extend(img_train_result)
        return final_analyze_train_result

if __name__ == "__main__":
    retinanet_model = RetinanetModel()
    modelush = retinanet_model
    res = modelush.validation_dataset('saraay@post.jce.ac.il')
    pass