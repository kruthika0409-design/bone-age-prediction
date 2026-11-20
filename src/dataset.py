import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocess import load_hand

# ResNet was trained on photos with these average colors, so we match them
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class BoneAgeDataset(Dataset):
    """Hands out (x-ray, sex, normalized age) one child at a time."""

    def __init__(self, df, img_dir, train=False, size=512):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.train = train
        self.size = size

    def __len__(self):
        return len(self.df)

    def augment(self, img):
        """Small random wiggles: turn, slide, brightness. img is float 0..1."""
        angle = np.random.uniform(-10, 10)                 # turn up to 10 degrees
        shift = self.size * 0.05                           # slide up to 5%
        dx = np.random.uniform(-shift, shift)
        dy = np.random.uniform(-shift, shift)

        center = (self.size / 2, self.size / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        M[0, 2] += dx
        M[1, 2] += dy
        img = cv2.warpAffine(img, M, (self.size, self.size), borderValue=0)

        brightness = np.random.uniform(0.8, 1.2)           # 20% darker or brighter
        return np.clip(img * brightness, 0.0, 1.0)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = os.path.join(self.img_dir, f"{int(row['id'])}.png")

        img = load_hand(path, self.size).astype(np.float32) / 255.0
        if self.train:
            img = self.augment(img)

        # ResNet wants 3 color layers, so we copy our gray picture 3 times
        img = np.stack([img, img, img], axis=0).astype(np.float32)
        img = (img - IMAGENET_MEAN) / IMAGENET_STD

        male = torch.tensor(float(row["male"]), dtype=torch.float32)
        target = torch.tensor(float(row["target"]), dtype=torch.float32)
        return torch.from_numpy(img), male, target