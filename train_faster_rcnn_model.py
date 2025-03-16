
import os
import torch
from typing import List
from helpers import Row, export_training_results_to_csv, draw_bounding_boxes_from_csv
from image_data import ImageData 
from torchvision import datasets 
from torch.utils.data import DataLoader
from detection_model import DetectionModel
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights, FasterRCNN


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_path = "models/faster_rcnn/fasterrcnn_resnet50.pth"
csv_results_path = "fasterrcnn_resnet50_results.csv"


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
    

class FasterRcnnModel(DetectionModel):
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
            self._model.roi_heads.nms_thresh = 0.2
            self._model.roi_heads.score_thresh = 0.5

            if os.path.exists(self.model_path):
                self._model.load_state_dict(torch.load(self.model_path, weights_only=True))
            else: 
                self.train_model(self._model)
        
        return self._model

    def train_model(self, model_to_train: FasterRCNN):
        train_dataset: datasets.ImageFolder = AncientScrollDataset('saraay@post.jce.ac.il')
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
        
        # Define optimizer and learning rate scheduler
        optimizer = torch.optim.SGD(model_to_train.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        
        # Training loop
        model_to_train.to(device)
        model_to_train.train()

        for epoch in range(5):
            self.train_one_epoch(model_to_train, optimizer, train_loader, device, epoch)

            # Save the model's state dictionary after every epoch
            torch.save(model_to_train.state_dict(), self.model_path)


    def train_one_epoch(self, model_to_train: FasterRCNN, optimizer, data_loader, device, epoch):
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

        

    def validation_dataset(self, validation_dir: str) -> List[Row]:
        dataset: datasets.ImageFolder = AncientScrollDataset(validation_dir)
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
    modelush = FasterRcnnModel()
    validation_imgs_dir = 'saraay@post.jce.ac.il/validate'
    csv_file = "FasterRcnnModel_validation_results.csv"
    # results = modelush.validation_dataset(validation_imgs_dir)
    # export_training_results_to_csv(
    #     csv_file=csv_file, 
    #     train_result=results
    # )
    draw_bounding_boxes_from_csv(validation_imgs_dir, csv_file)
    
