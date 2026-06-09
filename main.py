# import config
import config
import torch.optim as optim
from utils import (
mAP,
cells_to_bboxes,
get_evaluation_bboxes,
save_checkpoint,
load_checkpoint,
check_class_accuracy,
get_loaders,
plot_couple_examples
)
from tqdm import tqdm
from model import YOLOv3
from yoloLoss import YOLOLoss
import torch
import os

torch.backends.cudnn.benchmark = True

def train_fn(train_loaderm, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loaderm, leave=True)
    losses = []
    for batch_idxm, (x,y) in enumerate(loop):
        x = x.to(config.DEVICE)

        y0, y1, y2 = (
            y[0].to(config.DEVICE), y[1].to(config.DEVICE), y[2].to(config.DEVICE)
        )
        with torch.amp.autocast(device_type=config.DEVICE):
            output = model(x)
            loss = (
                loss_fn(output[0], y0, scaled_anchors[0])
                + loss_fn(output[1], y1, scaled_anchors[1])
                + loss_fn(output[2], y2, scaled_anchors[2])
            )

            losses.append(loss.item())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            mean_loss = sum(losses) / len(losses)
            loop.set_postfix(loss=mean_loss)

def main():
    print(config.DEVICE)
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimzer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fn = YOLOLoss()
    scaler = torch.amp.GradScaler(device=config.DEVICE)

    train_loader, test_loader, train_eval_loader = get_loaders(train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv",)

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_FILE, model, optimzer, config.LEARNING_RATE)


    scaled_anchors = (
        torch.tensor(config.ANCHORS) * torch.tensor(config.S).unsqueeze(1).unsqueeze(2).repeat(1, 3, 2)
    ).to(config.DEVICE)

    if config.LOAD_MODEL:
        pred_boxes, true_boxes = get_evaluation_bboxes(test_loader, model, iou_threshold=config.NMS_IOU_THRESH,
                                                       anchors=config.ANCHORS, threshold=config.CONF_THRESHOLD)
        mapval = mAP(pred_boxes, true_boxes, config.NUM_CLASSES, iou_thresh=config.MAP_IOU_THRESH, box_format="midpoint")
        print(f"MAP: {mapval}")
    for epoch in range(config.NUM_EPOCHS):


        train_fn(train_loader, model, optimzer, loss_fn, scaler, scaled_anchors)
        print("EPOCH", epoch)
        if config.SAVE_MODEL:
            save_checkpoint(model, optimzer)

        if epoch % 10 == 0 and epoch > 0:
            print("On Test Loader:")
            check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)

            pred_boxes, true_boxes = get_evaluation_bboxes(test_loader, model, iou_threshold=config.NMS_IOU_THRESH, anchors=config.ANCHORS, threshold=config.CONF_THRESHOLD)
            mapval = mAP(pred_boxes, true_boxes, config.NUM_CLASSES, iou_thresh=config.MAP_IOU_THRESH, box_format="midpoint")
            print(f"MAP: {mapval}")

if __name__ == "__main__":
    print("Run")
    main()
    print("DONE")