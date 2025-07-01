
import random
import torch
import torch.nn as nn
from utils import iou

class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10


    def forward(self, preds, targets, anchors):
        # No Object Loss
        obj = targets[..., 0] == 1
        noobj = targets[..., 0] == 0

        noobj_loss = self.bce((preds[..., 0:1][noobj]), (targets[..., 0:1][noobj]),)

        # Object Loss

        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat([self.sigmoid(preds[..., 1:3]), torch.exp(preds[..., 3:5]) * anchors], dim=-1)
        ious = iou(box_preds[obj], targets[..., 1:5][obj]).detach()
        object_loss = self.mse(self.sigmoid(preds[..., 0:1][obj]), ious * targets[..., 0:1][obj])

        # Boc Coords

        preds[..., 1:3] = self.sigmoid(preds[..., 1:3])  # x,y coordinates
        targets[..., 3:5] = torch.log(
            (1e-16 + targets[..., 3:5] / anchors)
        )  # width, height coordinates
        box_loss = self.mse(preds[..., 1:5][obj], targets[..., 1:5][obj])

        # Class Loss

        class_loss = self.ce(
            (preds[..., 5:][obj]), (targets[..., 5][obj].long()),
        )

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * noobj_loss
            + self.lambda_class * class_loss
        )

# import torch
# import torch.nn as nn
# from utils import iou
#
#
# class YOLOLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mse = nn.MSELoss()
#         self.bce = nn.BCEWithLogitsLoss()
#         self.ce = nn.CrossEntropyLoss()
#         self.sigmoid = nn.Sigmoid()
#
#         self.lambda_class = 1
#         self.lambda_noobj = 10
#         self.lambda_obj = 1
#         self.lambda_box = 10
#
#     def forward(self, targets, preds, anchors):
#         """
#         🔥 CRITICAL FIX: Fixed argument order and added validation
#         """
#         # Identify object and no-object masks
#         obj = targets[..., 0] == 1
#         noobj = targets[..., 0] == 0
#
#         # No Object Loss
#         noobj_loss = self.bce(preds[..., 0:1][noobj], targets[..., 0:1][noobj])
#
#         # Object Loss
#         anchors = anchors.reshape(1, 3, 1, 1, 2)
#         box_preds = torch.cat([
#             self.sigmoid(preds[..., 1:3]),
#             torch.exp(preds[..., 3:5]) * anchors
#         ], dim=-1)
#
#         # Only compute IoU if we have objects
#         if obj.sum() > 0:
#             ious = iou(box_preds[obj], targets[..., 1:5][obj]).detach()
#             object_loss = self.mse(self.sigmoid(preds[..., 0:1][obj]), ious * targets[..., 0:1][obj])
#
#             # Box Coordinates Loss
#             preds[..., 1:3] = self.sigmoid(preds[..., 1:3])  # x,y coordinates
#             targets[..., 3:5] = torch.log(1e-16 + targets[..., 3:5] / anchors)  # width, height
#             box_loss = self.mse(preds[..., 1:5][obj], targets[..., 1:5][obj])
#
#             # Class Loss - 🔥 CRITICAL FIX: Added validation
#             target_classes = targets[..., 5][obj].long()
#
#             # Validate class labels
#             if (target_classes < 0).any() or (target_classes >= preds.shape[-1] - 5).any():
#                 print(f"💥 Invalid class labels detected: {target_classes}")
#                 print(f"   Valid range: [0, {preds.shape[-1] - 6}]")
#                 # Clamp to valid range
#                 target_classes = torch.clamp(target_classes, 0, preds.shape[-1] - 6)
#
#             class_loss = self.ce(preds[..., 5:][obj], target_classes)
#         else:
#             # No objects in batch
#             object_loss = torch.tensor(0.0, device=targets.device, requires_grad=True)
#             box_loss = torch.tensor(0.0, device=targets.device, requires_grad=True)
#             class_loss = torch.tensor(0.0, device=targets.device, requires_grad=True)
#
#         return (
#                 self.lambda_box * box_loss
#                 + self.lambda_obj * object_loss
#                 + self.lambda_noobj * noobj_loss
#                 + self.lambda_class * class_loss
#         )