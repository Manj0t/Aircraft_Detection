from collections import Counter
import torch
import config
import os


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

def iou_width_height(boxes1, boxes2):
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union

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









def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("tab20b")
    class_labels = config.COCO_LABELS if config.DATASET=='COCO' else config.PASCAL_CLASSES
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=class_labels[int(class_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )

    plt.show()


def get_evaluation_bboxes(
    loader,
    model,
    iou_threshold,
    anchors,
    threshold,
    box_format="midpoint",
    device="cuda",
):
    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, (x, labels) in enumerate(tqdm(loader)):
        x = x.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device) * S
            boxes_scale_i = cells_to_bboxes(
                predictions[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        # we just want one bbox for each label, not one for each scale
        true_bboxes = cells_to_bboxes(
            labels[2], anchor, S=S, is_preds=False
        )

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    """
    Scales the predictions coming from the model to
    be relative to the entire image such that they for example later
    can be plotted or.
    INPUT:
    predictions: tensor of size (N, 3, S, S, num_classes+5)
    anchors: the anchors used for the predictions
    S: the number of cells the image is divided in on the width (and height)
    is_preds: whether the input is predictions or the true bounding boxes
    OUTPUT:
    converted_bboxes: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
                      object score, bounding box coordinates
    """
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
    return converted_bboxes.tolist()

def check_class_accuracy(model, loader, threshold):
    model.eval()
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0

    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.to(config.DEVICE)
        with torch.no_grad():
            out = model(x)

        for i in range(3):
            y[i] = y[i].to(config.DEVICE)
            obj = y[i][..., 0] == 1 # in paper this is Iobj_i
            noobj = y[i][..., 0] == 0  # in paper this is Iobj_i

            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
            )
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)

    print(f"Class accuracy is: {(correct_class/(tot_class_preds+1e-16))*100:2f}%")
    print(f"No obj accuracy is: {(correct_noobj/(tot_noobj+1e-16))*100:2f}%")
    print(f"Obj accuracy is: {(correct_obj/(tot_obj+1e-16))*100:2f}%")
    model.train()


def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        print("💾 No checkpoint found. Saving new one...")
        torch.save({
            'epoch': 0,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, checkpoint_file)

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_loaders(train_csv_path, test_csv_path):
    from dataset import YOLODataset
    from torch.utils.data import DataLoader

    IMAGE_SIZE = config.IMAGE_SIZE
    train_dataset = YOLODataset(
        train_csv_path,
        config.IMG_DIR,
        config.LABEL_DIR,
        config.ANCHORS,
        transform=config.train_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
    )
    test_dataset = YOLODataset(
        test_csv_path,
        config.IMG_DIR,
        config.LABEL_DIR,
        config.ANCHORS,
        transform=config.test_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    train_eval_dataset = YOLODataset(
        train_csv_path,
        config.IMG_DIR,
        config.LABEL_DIR,
        config.ANCHORS,
        transform=config.test_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
    )
    train_eval_loader = DataLoader(
        dataset=train_eval_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, test_loader, train_eval_loader

def plot_couple_examples(model, loader, thresh, iou_thresh, anchors):
    model.eval()
    x, y = next(iter(loader))
    x = x.to("cuda")
    with torch.no_grad():
        out = model(x)
        bboxes = [[] for _ in range(x.shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = anchors[i]
            boxes_scale_i = cells_to_bboxes(
                out[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        model.train()

    for i in range(batch_size):
        nms_boxes = non_max_suppression(
            bboxes[i], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
        )
        plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes)



def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False