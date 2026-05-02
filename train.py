import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from data_loader import BaseballPitchDataset

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train_model():
    device = torch.device('cpu')
    
    dataset_78 = BaseballPitchDataset(
        video_path=r'C:\Users\ben_k\OneDrive\Documents\GitHub\econ8310-assignment3-baseball\video_data\IMG_0078.mov', 
        xml_file=r'C:\Users\ben_k\OneDrive\Documents\GitHub\econ8310-assignment3-baseball\video_data\xml files\IMG_0078.xml'
    )
    
    dataset_79 = BaseballPitchDataset(
        video_path=r'C:\Users\ben_k\OneDrive\Documents\GitHub\econ8310-assignment3-baseball\video_data\IMG_0079.mov', 
        xml_file=r'C:\Users\ben_k\OneDrive\Documents\GitHub\econ8310-assignment3-baseball\video_data\xml files\IMG_0079.xml'
    )
    
    combined_dataset = ConcatDataset([dataset_78, dataset_79])
    
    data_loader = DataLoader(
        combined_dataset, 
        batch_size=2, 
        shuffle=True, 
        collate_fn=lambda x: tuple(zip(*x))
    )

    model = get_model(num_classes=2)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    model.train()
    num_epochs = 3 
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        
        for batch_idx, (images, targets) in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx} Loss: {losses.item():.4f}")
                
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss/len(data_loader):.4f}")

    torch.save(model.state_dict(), 'baseball_weights.pth')
    print("Done!")

train_model()