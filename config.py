import albumentations as A
import cv2
import torch

from albumentations.pytorch import ToTensorV2
from utils import seed_everything

DATASET = 'COCO'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# seed_everything()  # If you want deterministic behavior
NUM_WORKERS = 4
BATCH_SIZE = 8
IMAGE_SIZE = 416
NUM_CLASSES = 80
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
CONF_THRESHOLD = 0.4
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
PIN_MEMORY = True
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_FILE = "checkpoint.pth.tar"
IMG_DIR = DATASET + "/images/"
LABEL_DIR = DATASET + "/labels/"

ANCHORS =     [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]
#     [[(10,13), (16,30), (33,23)],
# [(30,61), (62,45), (59,119)],
# [(116,90), (156,198), (373,326)]]

# Note these have been rescaled to be between [0, 1]


scale = 1.1
train_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),

        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        A.OneOf(
            [
                A.ShiftScaleRotate(
                    rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                ),
                A.Compose([
                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
                    # other transforms...
                ])            ],
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.1),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
)
test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)



# scale = 1.05  # Slightly smaller than Pascal scale to reduce bbox out-of-bound risk
#
# train_transforms = A.Compose(
#     [
#         # Slight scale and pad to maintain aspect ratio
#         A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
#         A.PadIfNeeded(
#             min_height=int(IMAGE_SIZE * scale),
#             min_width=int(IMAGE_SIZE * scale),
#             border_mode=cv2.BORDER_CONSTANT,
#             value=(114, 114, 114)  # Same as YOLO padding color
#         ),
#
#         # Resize crop can distort YOLO coords, so keep tight control
#         A.RandomSizedBBoxSafeCrop(height=IMAGE_SIZE, width=IMAGE_SIZE, erosion_rate=0.0, p=0.5),
#
#         # Color + lighting augmentations
#         A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.4),
#
#         # Geometric but light to avoid pushing boxes out of bounds
#         A.OneOf(
#             [
#                 A.ShiftScaleRotate(
#                     shift_limit=0.05,
#                     scale_limit=0.05,
#                     rotate_limit=10,
#                     p=0.7,
#                     border_mode=cv2.BORDER_CONSTANT,
#                     value=(114, 114, 114),
#                 ),
#                 A.Affine(
#                     scale=(0.9, 1.1),
#                     translate_percent=0.05,
#                     rotate=(-10, 10),
#                     shear=(-2, 2),
#                     mode=cv2.BORDER_CONSTANT,
#                     cval=(114, 114, 114),
#                     p=0.7,
#                 ),
#             ],
#             p=0.7,
#         ),
#
#         # Flips and minor effects
#         A.HorizontalFlip(p=0.5),
#         A.GaussianBlur(blur_limit=(3, 5), p=0.1),
#         A.ToGray(p=0.05),
#         A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
#         ToTensorV2(),
#     ],
#     bbox_params=A.BboxParams(
#         format="yolo",
#         min_visibility=0.4,
#         label_fields=["class_labels"]
#     )
# )
#
# test_transforms = A.Compose(
#     [
#         A.LongestMaxSize(max_size=IMAGE_SIZE),
#         A.PadIfNeeded(
#             min_height=IMAGE_SIZE,
#             min_width=IMAGE_SIZE,
#             border_mode=cv2.BORDER_CONSTANT,
#             value=(114, 114, 114)
#         ),
#         A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
#         ToTensorV2(),
#     ],
#     bbox_params=A.BboxParams(
#         format="yolo",
#         min_visibility=0.4,
#         label_fields=["class_labels"]
#     )
# )

PASCAL_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

COCO_LABELS = [
 'person',
 'bicycle',
 'car',
 'motorcycle',
 'airplane',
 'bus',
 'train',
 'truck',
 'boat',
 'traffic light',
 'fire hydrant',
 'stop sign',
 'parking meter',
 'bench',
 'bird',
 'cat',
 'dog',
 'horse',
 'sheep',
 'cow',
 'elephant',
 'bear',
 'zebra',
 'giraffe',
 'backpack',
 'umbrella',
 'handbag',
 'tie',
 'suitcase',
 'frisbee',
 'skis',
 'snowboard',
 'sports ball',
 'kite',
 'baseball bat',
 'baseball glove',
 'skateboard',
 'surfboard',
 'tennis racket',
 'bottle',
 'wine glass',
 'cup',
 'fork',
 'knife',
 'spoon',
 'bowl',
 'banana',
 'apple',
 'sandwich',
 'orange',
 'broccoli',
 'carrot',
 'hot dog',
 'pizza',
 'donut',
 'cake',
 'chair',
 'couch',
 'potted plant',
 'bed',
 'dining table',
 'toilet',
 'tv',
 'laptop',
 'mouse',
 'remote',
 'keyboard',
 'cell phone',
 'microwave',
 'oven',
 'toaster',
 'sink',
 'refrigerator',
 'book',
 'clock',
 'vase',
 'scissors',
 'teddy bear',
 'hair drier',
 'toothbrush'
]