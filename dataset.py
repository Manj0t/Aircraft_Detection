# import os
# import torch
# from torch.utils.data import Dataset
# from PIL import Image, ImageFile
# from utils import iou_width_height as iou
# import numpy as np
# import pandas as pd
# import cv2
#
#
# ImageFile.LOAD_TRUNCATED_IMAGES = True
#
# class YOLODataset(Dataset):
#     def __init__(self, csv_file, image_dir, label_dir, anchors, num_classes=80, image_size=416, S=[13, 26, 52], transform=None):
#         self.annotations = pd.read_csv(csv_file)
#         self.img_dir = image_dir
#         self.label_dir = label_dir
#         self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
#         self.num_anchors = self.anchors.shape[0]
#         self.num_anchors_per_scale = self.num_anchors // 3
#         self.num_classes = num_classes
#         self.image_size = image_size
#         self.S = S
#         self.transform = transform
#         self.iou_thresh = 0.5
#
#     def __len__(self):
#         return len(self.annotations)
#
#     def __getitem__(self, idx, delimiter=" "):
#         img_filename = self.annotations.iloc[idx, 0]
#         label_filename = img_filename.replace(".jpg", ".txt")
#
#         label_path = os.path.join(self.label_dir, label_filename)
#         bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=delimiter, ndmin=2), 4, axis=1).tolist()
#         image_path = os.path.join(self.img_dir, img_filename)
#         image = np.array(Image.open(image_path).convert("RGB"))
#
#         # if self.transform:
#         #     for i in range(len(bboxes)):
#         #         for j in range(4):  # Only x_center, y_center, w, h
#         #             bboxes[i][j] = np.clip(bboxes[i][j], 0, 1)
#         #
#         #     augs = self.transform(image=image, bboxes=bboxes)
#         #     image = augs["image"]
#         #     bboxes = augs["bboxes"]
#
#         # Resize image to IMAGE_SIZE (e.g., 416x416)
#         image = cv2.resize(image, (self.image_size, self.image_size))
#         image = image / 255.0  # Normalize to [0, 1]
#         image = torch.tensor(image).permute(2, 0, 1).float()
#
#
#
#         targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
#         for box in bboxes:
#             x, y, width, height, class_label = box
#             class_label = int(class_label)
#
#             # 🔥 Check if the class label is invalid
#             if class_label < 0 or class_label >= self.num_classes:
#                 print(f"💥 Invalid class label: {class_label} in file: {label_filename}")
#                 raise ValueError(f"Class label {class_label} is out of range [0, {self.num_classes - 1}]")
#             class_label = int(box[4])
#             if class_label < 0 or class_label >= self.num_classes:
#                 raise ValueError(f"Invalid class label {class_label} found in file {label_path}")
#
#             iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
#             anchors_sorted_idx = iou_anchors.argsort(descending=True, dim=0)
#             x, y, width, height, class_label = box
#             has_anchor = [False] * 3
#             for idx in anchors_sorted_idx:
#                 scale_idx = idx // self.num_anchors_per_scale
#                 anchor_on_scale = idx % self.num_anchors_per_scale
#                 scale = self.S[scale_idx]
#                 i, j = int(scale * y), int(scale * x)
#                 anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
#                 if not anchor_taken and not has_anchor[scale_idx]:
#                     targets[scale_idx][anchor_on_scale, i, j, 0] = 1
#                     x_scaled, y_scaled = x * scale, y * scale
#                     width_scaled, height_scaled = width * scale, height * scale
#                     box_coords = torch.tensor([x_scaled, y_scaled, width_scaled, height_scaled])
#                     targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coords
#                     targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
#                     has_anchor[scale_idx] = True
#
#                 elif not anchor_taken and iou_anchors[idx] > self.iou_thresh:
#                     targets[scale_idx][anchor_on_scale, i, j, 0] = -1
#
#         return image, tuple(targets)


import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from utils import iou_width_height as iou
import numpy as np
import pandas as pd
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True


class YOLODataset(Dataset):
    def __init__(self, csv_file, image_dir, label_dir, anchors, num_classes=80, image_size=416, S=[13, 26, 52],
                 transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = image_dir
        self.label_dir = label_dir
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.num_classes = num_classes
        self.image_size = image_size
        self.S = S
        print('NUM CLASSES ', self.num_classes)
        self.transform = transform
        self.iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx, delimiter=" "):
        img_filename = self.annotations.iloc[idx, 0]
        label_filename = img_filename.replace(".jpg", ".txt")

        label_path = os.path.join(self.label_dir, label_filename)

        # Handle empty label files
        try:
            bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=delimiter, ndmin=2), 4, axis=1).tolist()
        except:
            print(f"Warning: Could not load labels from {label_path}")
            bboxes = []

        image_path = os.path.join(self.img_dir, img_filename)
        image = np.array(Image.open(image_path).convert("RGB"))

        # Resize image to IMAGE_SIZE (e.g., 416x416)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image / 255.0  # Normalize to [0, 1]
        image = torch.tensor(image).permute(2, 0, 1).float()

        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]

        for box in bboxes:
            x, y, width, height, class_label = box
            class_label = int(class_label)

            # 🔥 CRITICAL FIX: Validate class labels
            if class_label < 0 or class_label >= self.num_classes:
                print('NUM CLASSES AGAIN ', self.num_classes)

                print(f"💥 SKIPPING Invalid class label: {class_label} in file: {label_filename}")
                print(f"   Valid range is [0, {self.num_classes - 1}]")
                continue  # Skip this box instead of crashing

            # 🔥 CRITICAL FIX: Validate bounding box coordinates
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                print(f"💥 SKIPPING Invalid bbox coords: x={x}, y={y}, w={width}, h={height} in {label_filename}")
                continue  # Skip this box

            iou_anchors = iou(torch.tensor([width, height]), self.anchors)
            anchors_sorted_idx = iou_anchors.argsort(descending=True, dim=0)

            has_anchor = [False] * 3
            for anchor_idx in anchors_sorted_idx:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                scale = self.S[scale_idx]

                # 🔥 CRITICAL FIX: Ensure grid indices are valid
                i, j = int(scale * y), int(scale * x)
                i = min(i, scale - 1)  # Clamp to valid range
                j = min(j, scale - 1)  # Clamp to valid range

                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1

                    # Scale coordinates to grid
                    x_cell = scale * x - j  # Offset within cell
                    y_cell = scale * y - i  # Offset within cell
                    width_scaled = width * scale
                    height_scaled = height * scale

                    box_coords = torch.tensor([x_cell, y_cell, width_scaled, height_scaled])
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coords
                    targets[scale_idx][anchor_on_scale, i, j, 5] = class_label
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return image, tuple(targets)