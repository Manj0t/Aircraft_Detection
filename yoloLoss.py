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


    def forward(self, targets, preds, anchors):
        # No Object Loss
        obj = targets[..., 0] == 1
        noobj = targets[..., 0] == 0

        noobj_loss = self.bce((preds[..., 0:1][noobj]), (targets[..., 0:1][noobj]))

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

        class_loss = self.entropy(
            (preds[..., 5:][obj]), (targets[..., 5][obj].long()),
        )

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * noobj_loss
            + self.lambda_class * class_loss
        )