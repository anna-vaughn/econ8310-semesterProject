import os
import sys
import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from data_loader import BaseballPitchDataset
from pathlib import Path

# Store the locations of the baseball annotation and video files.
PROJECT_ROOT = Path(__file__).resolve().parents[0]
XML_DIR = PROJECT_ROOT / "Baseball Annotations"
VID_DIR = PROJECT_ROOT / "Baseball Videos"

# Throw an error if the folder locations can't be found.
if not XML_DIR.exists():
    raise FileNotFoundError(f"Could not find Baseball Annotations folder at: {XML_DIR}")

if not VID_DIR.exists():
    raise FileNotFoundError(f"Could not find Baseball Videos folder at: {VID_DIR}")

# Get all file names stored in the folders specified, and store them in a list.
vid_files = [os.path.join(VID_DIR, f) for f in os.listdir(VID_DIR) if f.endswith((".mov"))]
xml_files = [os.path.join(root, name) for root, dirs, files in os.walk(XML_DIR) for name in files if name.endswith((".xml"))]

# Get just filenames from the XML and video files.
xml_map = {os.path.splitext(os.path.basename(f))[0]: f for f in xml_files}
vid_map = {os.path.splitext(os.path.basename(f))[0]: f for f in vid_files}
# Sort the dictionaries and then parse for matching names.
match_names = sorted(set(xml_map.keys()) & set(vid_map.keys()))
pairs = [(vid_map[s], xml_map[s]) for s in match_names]
dropped = len(vid_files) - len(pairs)

# Check that the number of xml files matches the number of video files. If not, stop running this file.
if dropped == 0:
    print('Looking good, all files matched! Total number of files:', len(pairs))
else:
    print ('You are missing', dropped, 'videos! Add them to the appropriate folder and try running this file again.')
    sys.exit()

# Sepearate each image and target, Faster R-CNN expects thid format.
def collate_fn(batch):
    frames = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return frames, targets

# Split up the dataset for training, validation, and testing.
## First must sort our file names so we can ensure the match. 
## Then run our XML and video files through our custom dataset.
#s_vid_files = sorted(vid_files)
#s_xml_files = sorted(xml_files)
#full_ds = ConcatDataset([BaseballPitchDataset(x, y) for x, y in zip(s_vid_files, s_xml_files)])
full_ds = ConcatDataset([BaseballPitchDataset(vid, xml) for vid, xml in pairs])
n = len(full_ds)

# 80/20 split (train & val/test).
n_test = int(0.20 * n)
n_trainval = n - n_test

# 80/20 split (train/val), splits train set down further.
n_val = int(0.20 * n_trainval)
n_train = n_trainval - n_val

# Randomly split the files up into their respective datasets.
trainval_ds, test_ds = random_split(
    full_ds, [n_trainval, n_test],
    generator=torch.Generator().manual_seed(42))
train_ds, val_ds = random_split(
    trainval_ds, [n_train, n_val],
    generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=4, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=4, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=4, collate_fn=collate_fn)


# Build the model.
def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device('cpu')
# File name which will be used to store the weights for future use.
store_weights = "bball_frcnn.pth"

model = get_model(2).to(device)
print(model.roi_heads.box_predictor)

# Training the model:
EPOCHS = 2
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

def run_epoch(loader, train=True):
    model.train()
    total_loss = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, targets in loader:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
    return total_loss / len(loader)

for epoch in range(1, EPOCHS + 1):
    tr_loss = run_epoch(train_loader, train=True)
    val_loss = run_epoch(val_loader,   train=False)
    scheduler.step()
    print(f'Epoch {epoch:>2}/{EPOCHS}  Train Loss: {tr_loss:.4f}  Val Loss: {val_loss:.4f}')

# Save weights below as the file specified at the top of this code section.
torch.save(model.state_dict(), store_weights)