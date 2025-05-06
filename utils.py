from collections import Counter
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

def iou_matrix(box1, box2=None, format="corners"):
    if box2 is None:
        box2 = box1

    if format == "corners":
        box1_x1 = box1[:, 0]
        box1_y1 = box1[:, 1]
        box1_x2 = box1[:, 2]
        box1_y2 = box1[:, 3]

        box2_x1 = box2[:, 0]
        box2_y1 = box2[:, 1]
        box2_x2 = box2[:, 2]
        box2_y2 = box2[:, 3]

    elif format == "midpoint":
        box1_x1 = box1[:, 0] - box1[: , 2] / 2
        box1_y1 = box1[:, 1] - box1[: , 3] / 2
        box1_x2 = box1[: , 0] + box1[: , 2] / 2
        box1_y2 = box1[: , 1] + box1[: , 3] / 2
        box2_x1 = box2[:, 0] - box2[:, 2] / 2
        box2_y1 = box2[:, 1] - box2[:, 3] / 2
        box2_x2 = box2[:, 0] + box2[:, 2] / 2
        box2_y2 = box2[:, 1] + box2[:, 3] / 2
    else:
        raise Exception("format not supported")

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

    return ious

# Non max suppression

# May not be needed, will keep here commented out if issues arise with current nms later
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
    # Changes made, this sort may not be needed but will keep it commented out in case it is found that it is needed later on
    # pred = sorted(pred, key=lambda x: x[1], reverse=True)

    groups = {}

    for box in pred:
        cls = box[0]
        if cls not in groups:
            groups[cls] = []
        groups[cls].append(box)

    kept_boxes = []

    for boxes in groups.values():
        boxes = torch.tensor(boxes)

        predictions = boxes[:, 1]
        indices = predictions.argsort(descending=True)
        print(boxes[:, 2:])
        iou_mat = iou_matrix(boxes[:, 2:], format=format)
        keep = []

        while indices.numel() > 0:
            current = indices[0]
            keep.append(current.item())

            if indices.numel() == 1:
                break

            rest = indices[1:]
            curr_iou = iou_mat[current, rest]
            indices = rest[curr_iou <= iou_thresh]

        kept_boxes.append(boxes[keep])

    return torch.cat(kept_boxes).tolist() if kept_boxes else []


def mAP(pred_boxes, gt_boxes, num_classes, iou_thresh=0.5, box_format="corners"):
    # gt_boxes and pred_boxes = [[train_idx, class_pred, prob, x1, y1, x2, y2], ...]
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

    amt_boxes = Counter([gt[0] for gt in gt_boxes])

    for key, val in amt_boxes.items():
        amt_boxes[key] = torch.zeros(val)


    for c in range(num_classes):
        # If the class is not in detection_dict or gt_dict, skip loop
        if c not in detection_dict or c not in gt_dict:
            continue

        detections = detection_dict[c]
        # Sort by confidence score so we take the box with the higher confidence score with greater priority
        detections = sorted(detections, key=lambda x: x[2], reverse=True)
        gt_class = gt_dict[c]

        total_true_boxes = len(gt_class)


        FP = torch.zeros(len(detections))
        TP = torch.zeros(len(detections))

        for i, detection in enumerate(detections):
            curr_gts = []
            for gt_img in gt_class:
                if gt_img[0] == detection[0]:
                    curr_gts.append(gt_img)

            if len(curr_gts) == 0:
                FP[i] = 1
                continue

            curr_gts_tensor = torch.tensor([gts[3:] for gts in curr_gts])
            detection_tensor = torch.tensor(detection[3:])
            detection_tensor.unsqueeze_(0)

            iou_mat = iou_matrix(curr_gts_tensor, detection_tensor, format=box_format)

            max_iou, max_iou_idx = iou_mat.max(dim=0)
            max_iou = max_iou.item()
            max_iou_idx = max_iou_idx.item()

            if max_iou > iou_thresh and amt_boxes[detection[0]][max_iou_idx] == 0:
                TP[i] = 1
                amt_boxes[detection[0]][max_iou_idx] = 1
            else:
                FP[i] = 1

        TP = TP.cumsum(dim=0)
        FP = FP.cumsum(dim=0)

        recall = TP / (total_true_boxes + epsilon)
        precisions = torch.divide(TP, (TP + FP + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recall = torch.cat((torch.tensor([0]), recall))

        average_precisions.append(torch.trapz(precisions, recall))

    return sum(average_precisions) / (len(average_precisions) + epsilon)