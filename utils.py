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

# Non max suppression
def nms(pred, conf_thresh=0.5, iou_thresh=0.5, format="corners"):
    # pred: [[classNum, prediction, x1, y1, x2, y2], [...], [...]]
    assert type(pred) == list
    pred = sorted((p for p in pred if p[1] > conf_thresh), key=lambda x: x[1], reverse=True)

    found = {}
    for box in pred:
        if box[0] not in found:
            found[box[0]] = []
            found[box[0]].append(box)
        else:
            suppress = False
            for currBox in found[box[0]]:
                if iou(torch.tensor(currBox[2:]), torch.tensor(box[2:]), format) > iou_thresh:
                    suppress = True
                    break
            if not suppress:
                found[box[0]].append(box)

    return [box for boxes in found.values() for box in boxes]
