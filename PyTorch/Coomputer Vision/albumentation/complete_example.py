import torch
import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader


class CatDog(Dataset):
    def __init__(self, root_dir, transforms=None):
        super(CatDog, self).__init__()
        self.root_dir = root_dir
        self.transforms = transforms
        self.data = [(os.path.join(root_dir, "cats", path), 0) for path in os.listdir(os.path.join(root_dir, "cats"))]
        self.data += [(os.path.join(root_dir, "dogs", path), 1) for path in os.listdir(os.path.join(root_dir, "dogs"))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index]
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms:
            augmentations = self.transforms(image=image)
            augmented_image = augmentations["image"]

        return augmented_image, label

transform = A.Compose(
    [
        A.Resize(width=320, height=320),
        A.RandomCrop(width=200, height=200),
        A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
        A.OneOf([
                A.Blur(blur_limit=3, p=0.5),
                A.ColorJitter(p=0.5)
            ], p=0.1),
        A.Normalize(mean=[0,0,0],
                    std=[1,1,1],
                    max_pixel_value=255),
        ToTensorV2()
    ]
)

dataset = CatDog(root_dir="cat_dogs", transforms=transform)
loader = DataLoader(dataset, shuffle=True, batch_size=5)
for img, label in loader:
    print(img.shape, label.shape)
