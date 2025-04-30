import torch

def iou(target, pred, format="corners"):
    # Dataset only uses corners, ignore any other formats for this model
    if format == "corners":
        box1_x1 = target[..., 0:1]
        box1_y1 = target[..., 1:2]
        box1_x2 = target[..., 2:3]
        box1_y2 = target[..., 3:4]
        box2_x1 = pred[..., 0:1]
        box2_y1 = pred[..., 1:2]
        box2_x2 = pred[..., 2:3]
        box2_y2 = pred[..., 3:4]
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

def iou_matrix(boxes):
    x1 = boxes[:, 0].unsqueeze(1)
    y1 = boxes[:, 1].unsqueeze(1)
    x2 = boxes[:, 2].unsqueeze(1)
    y2 = boxes[:, 3].unsqueeze(1)

    xx1 = torch.max(x1, x1.T)
    yy1 = torch.max(y1, y1.T)
    xx2 = torch.min(x2, x2.T)
    yy2 = torch.min(y2, y2.T)

    intersection = (xx2 - xx1).clamp(0) * (yy2 - yy1).clamp(0)

    area = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    union = area + area.T - intersection

    ious = intersection / (union + 1e-6)
    ious.fill_diagonal_(0)

    return ious

# Non max suppression
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
            if keep_mask[i]:
                # Suppress boxes with IoU greater than the threshold
                iou_values = ious[i]
                suppress_mask = iou_values > iou_thresh
                keep_mask[suppress_mask] = False

        kept_boxes.append(boxes[keep_mask])

    return torch.cat(kept_boxes).tolist()

# Testing NMS

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

# Apply NMS to the test case
print("Original Predicted Boxes:")
for box in pred_boxes:
    print(box)

# Apply NMS
output_boxes = nms(pred_boxes, conf_thresh=0.7, iou_thresh=0.5)

print("\nAfter NMS (Non-Max Suppression):")
for box in output_boxes:
    print(box)