import os
import cv2
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from torchvision.io import read_image
from torchvision import tv_tensors

class BaseballPitchDataset(Dataset):
    def __init__(self, video_path, xml_file, transforms=None):
        self.transforms = transforms
        
        #Make a folder for the video frames
        self.img_dir = video_path.replace('.mov', '_frames').replace('.mp4', '_frames')
        
        #Make folder of frames exist if it doesn't
        if not os.path.exists(self.img_dir):
            self._extract_frames(video_path, self.img_dir)
        else:
            print(f"Frames already extracted here: {self.img_dir}.")

        #Parse XML
        self.frame_data = self._parse_cvat_xml(xml_file)
        self.valid_frames = sorted(list(self.frame_data.keys()))

    def _extract_frames(self, video_path, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break
            filepath = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(filepath, frame)
            frame_count += 1
            
        cap.release()

    def _parse_cvat_xml(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        frame_data = {}

        #Find baseballs
        for track in root.findall('track'):
            if track.get('label') == 'baseball':
                for box in track.findall('box'):
                    frame = int(box.get('frame'))
                    xtl = float(box.get('xtl'))
                    ytl = float(box.get('ytl'))
                    xbr = float(box.get('xbr'))
                    ybr = float(box.get('ybr'))

                    if frame not in frame_data:
                        frame_data[frame] = []
                    frame_data[frame].append([xtl, ytl, xbr, ybr])

        return frame_data

    def __getitem__(self, idx):
        frame_id = self.valid_frames[idx]
        
        #Load image
        img_name = f"frame_{frame_id:06d}.jpg" 
        img_path = os.path.join(self.img_dir, img_name)
        img = read_image(img_path)
        _, height, width = img.shape

        #Convert to float32
        img = img.to(torch.float32) / 255.0

        #Format boxes
        boxes = self.frame_data[frame_id]
        num_objs = len(boxes)
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        
        #0 is background, 1 is baseball
        labels = torch.ones((num_objs,), dtype=torch.int64) 
        
        area = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (boxes_tensor[:, 2] - boxes_tensor[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        img = tv_tensors.Image(img)
        
        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(
            boxes_tensor, format="XYXY", canvas_size=(height, width))
        target["labels"] = labels
        target["image_id"] = torch.tensor([frame_id], dtype=torch.int64)
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.valid_frames)