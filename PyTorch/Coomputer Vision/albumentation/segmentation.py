import cv2
import albumentations as A
import numpy as np
from utils import plot_examples
from PIL import Image

image = Image.open("images/elon.jpeg")
mask = Image.open("images/mask.jpeg")
mask2 = Image.open("images/second_mask.jpeg")

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
    ]
)

images = [image]
image = np.array(image)
mask = np.array(mask)
mask2 = np.array(mask2)
for i in range(4):
    augmentations = transform(image=image, masks=[mask, mask2])
    augmented_image = augmentations["image"]
    augmented_mask = augmentations["masks"]
    images.append(augmented_image)
    images.append(augmented_mask[0])
    images.append(augmented_mask[1])

plot_examples(images)
