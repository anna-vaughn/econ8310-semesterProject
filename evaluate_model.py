import torch
from pathlib import Path
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from train import split_data

# Loads in the trained model.
def load_trained_model(weights_path, num_classes=2):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval() 
    return model

# Evaluate the model using our testing data. This runs over every batch and calculates the accuracy.
def evaluate(model: torch.nn.Module, loader, confidence_threshold: float = 0.5):
    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device('cpu') 
    metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy")
 
    with torch.no_grad():
        for batch_idx, (imgs, targets) in enumerate(loader, start=1):
 
            # Move images and targets to the correct device.
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
 
            # Run inference.
            predictions = model(imgs)
 
            # Filter out low-confidence predictions before scoring.
            filtered_preds = []
            for pred in predictions:
                keep = pred["scores"] >= confidence_threshold
                filtered_preds.append({
                    "boxes": pred["boxes"][keep],
                    "scores": pred["scores"][keep],
                    "labels": pred["labels"][keep],
                })

            metric.update(filtered_preds, targets)
 
    # Compute all metrics at once.
    results = metric.compute()
    return results


# Evaluate the model on our testing data.
PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = PROJECT_ROOT / "bball_frcnn.pth"
confidence = 0.5

if not WEIGHTS_PATH.exists():
    raise FileNotFoundError(f"No weights file found at {WEIGHTS_PATH}.\nDownload bball_frcnn.pth from the Google Drive link in README.md and add it to the main project folder.")

print(f"\nLoading model weights from: {WEIGHTS_PATH}")
model = load_trained_model(str(WEIGHTS_PATH))

# Placeholder for train and val loaders as we're only going to use test, since split_data() returns a tuple. 
_, _, test_loader = split_data()

print(f"\nRunning evaluation over {len(test_loader)} batches...\n")
results = evaluate(model, test_loader, confidence_threshold=confidence)

# Get the two metrics we care about for our model from the results.
map_val = results["map"].item()
map50 = results["map_50"].item()

print(f"Confidence threshold: {confidence}")
print(f"Test set size: {len(test_loader.dataset)} frames")

print("\n=== Metrics ===")
print(f"mAP (IoU 0.50–0.95): {map_val:.4f}")
print(f"mAP@50: {map50:.4f}")