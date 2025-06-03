import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from utils import iou_width_height as iou
import numpy as np
import pandas as pd


ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    def __init__(self, csv_file, image_dir, label_dir, anchors, num_classes=20, image_size=416, S=[13, 26, 52], transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = image_dir
        self.label_dir = label_dir
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.num_classes = num_classes
        self.image_size = image_size
        self.S = S
        self.transform = transform
        self.iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx, delimiter=" "):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[idx, 0])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=delimiter, ndmin=2), 4, axis=1).tolist()
        image_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = np.array(Image.open(image_path).convert("RGB"))

        if self.transform:
            augs = self.transform(image=image, bboxes=bboxes)
            image = augs["image"]
            bboxes = augs["bboxes"]

        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchors_sorted_idx = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3
            for idx in anchors_sorted_idx:
                scale_idx = idx // self.num_anchors_per_scale
                anchor_on_scale = idx % self.num_anchors_per_scale
                scale = self.S[scale_idx]
                i, j = int(scale * y), int(scale * x)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_scaled, y_scaled = x * scale, y * scale
                    width_scaled, height_scaled = width * scale, height * scale
                    box_coords = torch.tensor([x_scaled, y_scaled, width_scaled, height_scaled])
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coords
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[idx] > self.iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return image, tuple(targets)