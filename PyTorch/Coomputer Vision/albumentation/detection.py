import cv2
import albumentations as A
import numpy as np
from utils import plot_examples
from PIL import Image

image = cv2.imread('images/cat.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

bboxes = [[13, 170, 240, 410]]

transform = A.Compose(
    [
        A.Resize(width=1920, height=1080),
        A.RandomCrop(width=1280, height=720),
        A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
        A.OneOf([
                A.Blur(blur_limit=3, p=0.5),
                A.ColorJitter(p=0.5)
            ], p=0.1)
    ],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=[])
)

images = [image]
saved_bboxes = [bboxes[0]]
for i in range(15):
    augmentations = transform(image=image, bboxes=bboxes)
    augmented_image = augmentations["image"]
    images.append(augmented_image)
    saved_bboxes.append(augmentations["bboxes"][0])

plot_examples(images, saved_bboxes)
