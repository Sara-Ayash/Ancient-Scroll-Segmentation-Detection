
import os
import torch
from typing import List
from torchvision import datasets 
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

from ancient_scroll_dataset import AncientScrollDataset, ImageData, Row
from helpers import export_training_results_to_csv, train_one_epoch 

csv_file="fasterrcnn_resnet50_results.csv"
fasterrcnn_model_path = "fasterrcnn_resnet50.pth"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



def get_fasterrcnn_model(fasterrcnn_model_path):
    # Load the pre-trained Faster R-CNN model with a ResNet-50 backbone
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
 
    if os.path.exists(fasterrcnn_model_path):
        model.load_state_dict(torch.load(fasterrcnn_model_path, weights_only=True))
    
    return model


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