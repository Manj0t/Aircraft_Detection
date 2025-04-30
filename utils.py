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

    iou = intersection / (union + 1e-6)
    iou.fill_diagonal_(0)
    iou = torch.max(iou, dim=1)

    return iou

# Non max suppression
def nms(pred, conf_thresh=0.5, iou_thresh=0.5, format="corners"):
    # pred: [[classNum, prediction, x1, y1, x2, y2], [...], [...]]
    assert type(pred) == list
    pred = (p for p in pred if p[1] > conf_thresh)
    pred = sorted(pred, key=lambda x: x[1], reverse=True)

    boxes = []

    for box in pred:

    # found = {}
    # for box in pred:
    #     if box[0] not in found:
    #         found[box[0]] = []
    #         found[box[0]].append(box)
    #     else:
    #         suppress = False
    #         for currBox in found[box[0]]:
    #             if iou(torch.tensor(currBox[2:]), torch.tensor(box[2:]), format) > iou_thresh:
    #                 suppress = True
    #                 break
    #         if not suppress:
    #             found[box[0]].append(box)

    return [box for boxes in found.values() for box in boxes]
