import torch

def iou(target, pred, format="corners"):
    if format == "corners":
        box1_x1 = target[..., 0:1]
        box1_y1 = target[..., 1:2]
        box1_x2 = target[..., 2:3]
        box1_y2 = target[..., 3:4]
        box2_x1 = pred[..., 0:1]
        box2_y1 = pred[..., 1:2]
        box2_x2 = pred[..., 2:3]
        box2_y2 = pred[..., 3:4]
    elif format == "midpoint":
        box1_x1 = pred[..., 0:1] - pred[..., 2:3] / 2
        box1_y1 = pred[..., 1:2] - pred[..., 3:4] / 2
        box1_x2 = pred[..., 0:1] + pred[..., 2:3] / 2
        box1_y2 = pred[..., 1:2] + pred[..., 3:4] / 2
        box2_x1 = target[..., 0:1] - target[..., 2:3] / 2
        box2_y1 = target[..., 1:2] - target[..., 3:4] / 2
        box2_x2 = target[..., 0:1] + target[..., 2:3] / 2
        box2_y2 = target[..., 1:2] + target[..., 3:4] / 2
    else:
        raise Exception("format not supported")

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    union = box1_area + box2_area - intersection

    return intersection / (union + 1e-6)

def iou_matrix(box1, box2=None):
    if box2 is None:
        box2 = box1
    box1_x1 = box1[:, 0]
    box1_y1 = box1[:, 1]
    box1_x2 = box1[:, 2]
    box1_y2 = box1[:, 3]

    box2_x1 = box2[:, 0]
    box2_y1 = box2[:, 1]
    box2_x2 = box2[:, 2]
    box2_y2 = box2[:, 3]

    A = box1.shape[0]
    B = box2.shape[0]

    box1_x1 = box1_x1.view(A, 1)
    box1_y1 = box1_y1.view(A, 1)
    box1_x2 = box1_x2.view(A, 1)
    box1_y2 = box1_y2.view(A, 1)

    box2_x1 = box2_x1.view(1, B)
    box2_y1 = box2_y1.view(1, B)
    box2_x2 = box2_x2.view(1, B)
    box2_y2 = box2_y2.view(1, B)

    xx1 = torch.max(box1_x1, box2_x1)
    yy1 = torch.max(box1_y1, box2_y1)
    xx2 = torch.min(box1_x2, box2_x2)
    yy2 = torch.min(box1_y2, box2_y2)

    intersection = (xx2 - xx1).clamp(0) * (yy2 - yy1).clamp(0)

    area1 = (box1_x2 - box1_x1).clamp(0) * (box1_y2 - box1_y1).clamp(0)
    area2 = (box2_x2 - box2_x1).clamp(0) * (box2_y2 - box2_y1).clamp(0)
    union = area1 + area2 - intersection

    ious = intersection / (union + 1e-6)
    if box1.data_ptr() == box2.data_ptr():
        ious.fill_diagonal_(0)

    return ious

# Non max suppression
# def nms(pred, conf_thresh=0.5, iou_thresh=0.5, format="corners"):
#     # pred: [[classNum, prediction, x1, y1, x2, y2], [...], [...]]
#     assert type(pred) == list
#     pred = [box for box in pred if box[1] > conf_thresh]
#     pred = sorted(pred, key=lambda x: x[1], reverse=True)
#
#     found = {}
#     for box in pred:
#         if box[0] not in found:
#             found[box[0]] = []
#             found[box[0]].append(box)
#         else:
#             suppress = False
#             for currBox in found[box[0]]:
#                 if iou(torch.tensor(currBox[2:]).unsqueeze(0), torch.tensor(box[2:]).unsqueeze(0), format) >= iou_thresh:
#                     suppress = True
#                     break
#             if not suppress:
#                 found[box[0]].append(box)
#
#     return [box for boxes in found.values() for box in boxes]


def nms(pred, conf_thresh=0.5, iou_thresh=0.5, format="corners"):
    # pred: [[classNum, prediction, x1, y1, x2, y2], [...], [...]]
    assert type(pred) == list
    pred = [p for p in pred if p[1] > conf_thresh]
    # sorts by class and confidence assuming there are < 1000 classes
    pred = sorted(pred, key=lambda x: x[1], reverse=True)

    groups = {}

    for box in pred:
        cls = box[0]
        if cls not in groups:
            groups[cls] = []
        groups[cls].append(box)

    kept_boxes = []

    for boxes in groups.values():
        boxes = torch.tensor(boxes)
        ious = iou_matrix(boxes[:, 2:])

        keep_mask = torch.ones(boxes.shape[0], dtype=torch.bool)
        for i in range(len(boxes)):
            # Suppress boxes with IoU greater than the threshold
            iou_values = ious[i]
            suppress_mask = iou_values > iou_thresh
            keep_mask[suppress_mask] = False
        kept_boxes.append(boxes[keep_mask])

    return torch.cat(kept_boxes).tolist()


def mAP(pred_boxes, gt_boxes, num_classes, iou_thresh=0.5, box_format="corners"):
        # pred_boxes = [[train_idx, class_pred, prob, x1, y1, x2, y2], ...]
    average_precisions = []
    epsilon = 1e-6

    detection_dict  = {}
    gt_dict = {}
    for box in pred_boxes:
        if box[1] not in detection_dict:
            detection_dict[box[1]] = []
        detection_dict[box[1]].append(box)
    for box in gt_boxes:
        if box[1] not in gt_dict:
            gt_dict[box[1]] = []
        gt_dict[box[1]].append(box)

    for c in range(num_classes):
        detections = detection_dict[c]
        class_gt = gt_dict[c]

        detections = torch.tensor(detections)
        class_gt = torch.tensor(class_gt)

        ious = iou_matrix(detections[:, 3:], class_gt[:, 3:])
        FP_maks = ious < iou_thresh







pred_boxes = [
    # Class 0 (Aircraft)
    [0, 0.92, 50, 50, 150, 150],  # High confidence, good prediction
    [0, 0.75, 60, 60, 160, 160],  # Overlapping with the first one
    [0, 0.65, 55, 55, 155, 155],  # Lower confidence, overlapping with the first two
    [0, 0.80, 80, 80, 180, 180],  # Overlapping, but still confident
    [0, 0.30, 500, 500, 600, 600],  # Low confidence, far from other boxes

    # Class 1 (Helicopter)
    [1, 0.85, 200, 200, 300, 300],  # Confident, non-overlapping
    [1, 0.90, 210, 210, 310, 310],  # High confidence, overlapping with the first one
    [1, 0.40, 600, 600, 700, 700],  # Low confidence, far from others
    [1, 0.78, 230, 230, 330, 330],  # Overlapping with the first one, decent confidence

    # Class 2 (Car)
    [2, 0.60, 100, 100, 250, 250],  # Overlapping with other boxes
    [2, 0.82, 200, 200, 400, 400],  # High confidence, some overlap
    [2, 0.50, 300, 300, 500, 500],  # Low confidence
    [2, 0.95, 350, 350, 550, 550],  # Very high confidence, overlapping
    [2, 0.55, 100, 100, 150, 150],  # Overlapping with first car box

    # Class 3 (Bicycle)
    [3, 0.80, 300, 300, 450, 450],  # Confident, overlapping with car
    [3, 0.75, 350, 350, 500, 500],  # Decent confidence, overlap
    [3, 0.60, 400, 400, 600, 600],  # Low confidence
    [3, 0.90, 450, 450, 650, 650],  # High confidence, overlaps with other bikes
    [3, 0.88, 500, 500, 700, 700],  # High confidence, far from others
]

# IoU and NMS Testing
print("Original Predicted Boxes:")
for box in pred_boxes:
    print(box)

# Apply NMS
output_boxes = nms(pred_boxes, conf_thresh=0.7, iou_thresh=0.5)

print("\nAfter NMS (Non-Max Suppression):")
for box in output_boxes:
    print(box)

#     # [2, 0.95, 350, 350, 550, 550]
#     # [0, 0.92, 50, 50, 150, 150]
#     # [1, 0.9, 210, 210, 310, 310]
#     # [3, 0.9, 450, 450, 650, 650]
#     # [3, 0.88, 500, 500, 700, 700]
#     [1, 0.85, 200, 200, 300, 300]
#     # [2, 0.82, 200, 200, 400, 400]
#     # [0, 0.8, 80, 80, 180, 180]
#     # [3, 0.8, 300, 300, 450, 450]
#     # [1, 0.78, 230, 230, 330, 330]
#     [0, 0.75, 60, 60, 160, 160]
#     # [3, 0.75, 350, 350, 500, 500]
#     [2, 0.6, 100, 100, 250, 250]
#     [3, 0.6, 400, 400, 600, 600]
#     [2, 0.55, 100, 100, 150, 150]
#
#     # [2, 0.95, 350, 350, 550, 550]
#     # [2, 0.82, 200, 200, 400, 400]
#     # [0, 0.92, 50, 50, 150, 150]
#     # [0, 0.8, 80, 80, 180, 180]
#     # [1, 0.9, 210, 210, 310, 310]
#     # [1, 0.78, 230, 230, 330, 330]
#     # [3, 0.9, 450, 450, 650, 650]
#     # [3, 0.88, 500, 500, 700, 700]
#     # [3, 0.8, 300, 300, 450, 450]
#     # [3, 0.75, 350, 350, 500, 500]
#
#
# [2, 0.95, 350, 350, 550, 550]
# [0, 0.92, 50, 50, 150, 150]
# [0, 0.75, 60, 60, 160, 160]
# [1, 0.9, 210, 210, 310, 310]
# [1, 0.85, 200, 200, 300, 300]
# [3, 0.9, 450, 450, 650, 650]
